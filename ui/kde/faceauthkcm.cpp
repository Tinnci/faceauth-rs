#include "faceauthkcm.h"

#include <KPluginFactory>

#include <QDBusConnection>
#include <QDBusError>
#include <QDBusInterface>
#include <QDBusPendingCallWatcher>
#include <QDBusPendingReply>
#include <QDBusServiceWatcher>
#include <QUuid>

#include <unistd.h>

namespace
{
constexpr auto managerService = "org.faceauth.Manager1";
constexpr auto managerPath = "/org/faceauth/Manager1";
constexpr auto managerInterface = "org.faceauth.Manager1";
constexpr quint16 supportedSchemaVersion = 2;

bool isCanonicalOperationId(const QString &value)
{
    const auto id = QUuid::fromString(value);
    return !id.isNull() && id.toString(QUuid::WithoutBraces) == value;
}

bool isKnownProgress(const QString &progress)
{
    return progress == QLatin1String("preparing") || progress == QLatin1String("position-face")
        || progress == QLatin1String("hold-still")
        || progress == QLatin1String("active-challenge") || progress == QLatin1String("blink")
        || progress == QLatin1String("turn-left") || progress == QLatin1String("turn-right")
        || progress == QLatin1String("return-to-center")
        || progress == QLatin1String("processing");
}

bool isKnownResult(const QString &result)
{
    return result == QLatin1String("completed") || result == QLatin1String("cancelled")
        || result == QLatin1String("timed-out") || result == QLatin1String("failed");
}

QString experienceOutcome(const QString &result)
{
    if (result == QLatin1String("completed")) {
        return QStringLiteral("succeeded");
    }
    if (result == QLatin1String("failed")) {
        return QStringLiteral("unavailable");
    }
    return result;
}

QString errorCodeFor(const QDBusError &error, const QString &fallback)
{
    if (error.type() == QDBusError::AccessDenied) {
        return QStringLiteral("authorization-failed");
    }
    if (error.type() == QDBusError::NoReply || error.type() == QDBusError::ServiceUnknown) {
        return QStringLiteral("daemon-unavailable");
    }
    return fallback;
}
}

K_PLUGIN_CLASS_WITH_JSON(FaceAuthKcm, "kcm_faceauth.json")

FaceAuthKcm::FaceAuthKcm(QObject *parent, const KPluginMetaData &metaData)
    : KQuickConfigModule(parent, metaData)
    , m_manager(new QDBusInterface(QString::fromLatin1(managerService),
                                   QString::fromLatin1(managerPath),
                                   QString::fromLatin1(managerInterface),
                                   QDBusConnection::systemBus(),
                                   this))
    , m_serviceWatcher(new QDBusServiceWatcher(QString::fromLatin1(managerService),
                                               QDBusConnection::systemBus(),
                                               QDBusServiceWatcher::WatchForRegistration
                                                   | QDBusServiceWatcher::WatchForUnregistration,
                                               this))
    , m_uid(static_cast<quint32>(geteuid()))
{
    auto bus = QDBusConnection::systemBus();
    bus.connect(QString::fromLatin1(managerService),
                QString::fromLatin1(managerPath),
                QString::fromLatin1(managerInterface),
                QStringLiteral("EnrollmentProgress"),
                this,
                SLOT(handleEnrollmentProgress(QString,QString)));
    bus.connect(QString::fromLatin1(managerService),
                QString::fromLatin1(managerPath),
                QString::fromLatin1(managerInterface),
                QStringLiteral("EnrollmentCompleted"),
                this,
                SLOT(handleEnrollmentCompleted(QString,QString)));
    connect(m_serviceWatcher,
            &QDBusServiceWatcher::serviceRegistered,
            this,
            &FaceAuthKcm::handleServiceRegistered);
    connect(m_serviceWatcher,
            &QDBusServiceWatcher::serviceUnregistered,
            this,
            &FaceAuthKcm::handleServiceUnregistered);
    refresh();
}

FaceAuthKcm::~FaceAuthKcm() = default;

bool FaceAuthKcm::daemonAvailable() const
{
    return m_daemonAvailable;
}

bool FaceAuthKcm::enrolled() const
{
    return m_enrolled;
}

bool FaceAuthKcm::busy() const
{
    return m_starting || !m_operationId.isEmpty();
}

quint16 FaceAuthKcm::schemaVersion() const
{
    return m_schemaVersion;
}

QString FaceAuthKcm::cue() const
{
    return m_cue;
}

QString FaceAuthKcm::outcome() const
{
    return m_outcome;
}

QString FaceAuthKcm::errorCode() const
{
    return m_errorCode;
}

void FaceAuthKcm::refresh()
{
    const auto generation = ++m_generation;
    setErrorCode({});
    refreshVersion(generation);
    refreshEnrollmentState(generation);
}

void FaceAuthKcm::refreshVersion(quint64 generation)
{
    auto *watcher = new QDBusPendingCallWatcher(m_manager->asyncCall(QStringLiteral("GetVersion")), this);
    connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, watcher, generation]() {
        const QDBusPendingReply<quint16> reply = *watcher;
        watcher->deleteLater();
        if (generation != m_generation) {
            return;
        }
        if (reply.isError()) {
            setDaemonAvailable(false);
            setSchemaVersion(0);
            setErrorCode(errorCodeFor(reply.error(), QStringLiteral("request-failed")));
            return;
        }
        setDaemonAvailable(true);
        setSchemaVersion(reply.value());
        if (reply.value() != supportedSchemaVersion) {
            setErrorCode(QStringLiteral("protocol-mismatch"));
        }
    });
}

void FaceAuthKcm::refreshEnrollmentState(quint64 generation)
{
    auto *watcher = new QDBusPendingCallWatcher(
        m_manager->asyncCall(QStringLiteral("GetEnrollmentState"), QVariant::fromValue(m_uid)), this);
    connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, watcher, generation]() {
        const QDBusPendingReply<bool> reply = *watcher;
        watcher->deleteLater();
        if (generation != m_generation) {
            return;
        }
        if (reply.isError()) {
            setDaemonAvailable(false);
            setErrorCode(errorCodeFor(reply.error(), QStringLiteral("request-failed")));
            return;
        }
        setDaemonAvailable(true);
        setEnrolled(reply.value());
    });
}

void FaceAuthKcm::beginEnrollment()
{
    if (busy() || !m_daemonAvailable || m_schemaVersion != supportedSchemaVersion) {
        return;
    }
    const auto generation = ++m_generation;
    setStarting(true);
    setCue(QStringLiteral("preparing"));
    setOutcome({});
    setErrorCode({});
    auto *watcher = new QDBusPendingCallWatcher(
        m_manager->asyncCall(QStringLiteral("BeginEnrollment"), QVariant::fromValue(m_uid)), this);
    connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, watcher, generation]() {
        const QDBusPendingReply<QString> reply = *watcher;
        watcher->deleteLater();
        if (generation != m_generation) {
            return;
        }
        setStarting(false);
        if (reply.isError()) {
            setCue({});
            setErrorCode(errorCodeFor(reply.error(), QStringLiteral("enrollment-failed")));
            return;
        }
        if (!isCanonicalOperationId(reply.value())
            || (!m_operationId.isEmpty() && m_operationId != reply.value())) {
            setOperationId({});
            setCue({});
            setErrorCode(QStringLiteral("invalid-operation"));
            return;
        }
        setOperationId(reply.value());
    });
}

void FaceAuthKcm::cancelEnrollment()
{
    if (m_operationId.isEmpty()) {
        return;
    }
    const auto generation = m_generation;
    auto *watcher = new QDBusPendingCallWatcher(
        m_manager->asyncCall(QStringLiteral("CancelEnrollment"),
                             QVariant::fromValue(m_uid),
                             m_operationId),
        this);
    connect(watcher, &QDBusPendingCallWatcher::finished, this, [this, watcher, generation]() {
        const QDBusPendingReply<> reply = *watcher;
        watcher->deleteLater();
        if (generation != m_generation) {
            return;
        }
        if (reply.isError()) {
            setErrorCode(errorCodeFor(reply.error(), QStringLiteral("cancel-failed")));
        }
    });
}

void FaceAuthKcm::handleServiceRegistered()
{
    refresh();
}

void FaceAuthKcm::handleServiceUnregistered()
{
    ++m_generation;
    setDaemonAvailable(false);
    setSchemaVersion(0);
    setStarting(false);
    setOperationId({});
    setCue({});
    setOutcome(QStringLiteral("unavailable"));
    setErrorCode(QStringLiteral("daemon-unavailable"));
}

void FaceAuthKcm::handleEnrollmentProgress(const QString &operationId, const QString &progress)
{
    if (!isCanonicalOperationId(operationId) || !isKnownProgress(progress)) {
        setErrorCode(QStringLiteral("protocol-mismatch"));
        return;
    }
    if (m_operationId.isEmpty() && m_starting) {
        setOperationId(operationId);
    }
    if (operationId != m_operationId) {
        return;
    }
    setCue(progress);
}

void FaceAuthKcm::handleEnrollmentCompleted(const QString &operationId, const QString &result)
{
    if (!isCanonicalOperationId(operationId)) {
        setErrorCode(QStringLiteral("protocol-mismatch"));
        return;
    }
    if (m_operationId.isEmpty() && m_starting) {
        setOperationId(operationId);
    }
    if (operationId != m_operationId) {
        return;
    }
    ++m_generation;
    setStarting(false);
    setCue({});
    if (isKnownResult(result)) {
        setOutcome(experienceOutcome(result));
    } else {
        setOutcome(QStringLiteral("unavailable"));
        setErrorCode(QStringLiteral("protocol-mismatch"));
    }
    setOperationId({});
    if (result == QLatin1String("completed")) {
        setEnrolled(true);
    }
}

void FaceAuthKcm::setDaemonAvailable(bool available)
{
    if (m_daemonAvailable == available) {
        return;
    }
    m_daemonAvailable = available;
    Q_EMIT daemonAvailableChanged();
}

void FaceAuthKcm::setEnrolled(bool enrolled)
{
    if (m_enrolled == enrolled) {
        return;
    }
    m_enrolled = enrolled;
    Q_EMIT enrolledChanged();
}

void FaceAuthKcm::setStarting(bool starting)
{
    if (m_starting == starting) {
        return;
    }
    const bool wasBusy = busy();
    m_starting = starting;
    if (wasBusy != busy()) {
        Q_EMIT busyChanged();
    }
}

void FaceAuthKcm::setSchemaVersion(quint16 version)
{
    if (m_schemaVersion == version) {
        return;
    }
    m_schemaVersion = version;
    Q_EMIT schemaVersionChanged();
}

void FaceAuthKcm::setCue(const QString &cue)
{
    if (m_cue == cue) {
        return;
    }
    m_cue = cue;
    Q_EMIT cueChanged();
}

void FaceAuthKcm::setOutcome(const QString &outcome)
{
    if (m_outcome == outcome) {
        return;
    }
    m_outcome = outcome;
    Q_EMIT outcomeChanged();
}

void FaceAuthKcm::setErrorCode(const QString &errorCode)
{
    if (m_errorCode == errorCode) {
        return;
    }
    m_errorCode = errorCode;
    Q_EMIT errorCodeChanged();
}

void FaceAuthKcm::setOperationId(const QString &operationId)
{
    if (m_operationId == operationId) {
        return;
    }
    const bool wasBusy = busy();
    m_operationId = operationId;
    if (wasBusy != busy()) {
        Q_EMIT busyChanged();
    }
}

#include "faceauthkcm.moc"
