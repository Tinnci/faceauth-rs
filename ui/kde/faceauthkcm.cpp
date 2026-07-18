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

QString uniqueBusConnectionName()
{
    return QStringLiteral("faceauth-kcm-")
        + QUuid::createUuid().toString(QUuid::WithoutBraces);
}

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
    , m_busConnectionName(uniqueBusConnectionName())
    , m_uid(static_cast<quint32>(geteuid()))
{
    auto bus = QDBusConnection::connectToBus(QDBusConnection::SystemBus, m_busConnectionName);
    m_manager = new QDBusInterface(QString::fromLatin1(managerService),
                                   QString::fromLatin1(managerPath),
                                   QString::fromLatin1(managerInterface),
                                   bus,
                                   this);
    m_serviceWatcher = new QDBusServiceWatcher(
        QString::fromLatin1(managerService),
        bus,
        QDBusServiceWatcher::WatchForRegistration | QDBusServiceWatcher::WatchForUnregistration,
        this);
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

FaceAuthKcm::~FaceAuthKcm()
{
    bestEffortCancel();
    QDBusConnection::disconnectFromBus(m_busConnectionName);
}

bool FaceAuthKcm::daemonAvailable() const
{
    return m_daemonAvailable;
}

bool FaceAuthKcm::enrolled() const
{
    return m_enrolled;
}

bool FaceAuthKcm::loading() const
{
    return m_pageState == PageState::Loading;
}

bool FaceAuthKcm::busy() const
{
    return m_pageState == PageState::Authorizing || m_pageState == PageState::Starting
        || m_pageState == PageState::Capturing || m_pageState == PageState::Cancelling;
}

bool FaceAuthKcm::canCancel() const
{
    return !m_operationId.isEmpty()
        && (m_pageState == PageState::Starting || m_pageState == PageState::Capturing);
}

QString FaceAuthKcm::pageState() const
{
    switch (m_pageState) {
    case PageState::Loading:
        return QStringLiteral("loading");
    case PageState::Ready:
        return QStringLiteral("ready");
    case PageState::Authorizing:
        return QStringLiteral("authorizing");
    case PageState::Starting:
        return QStringLiteral("starting");
    case PageState::Capturing:
        return QStringLiteral("capturing");
    case PageState::Cancelling:
        return QStringLiteral("cancelling");
    case PageState::Terminal:
        return QStringLiteral("terminal");
    }
    Q_UNREACHABLE_RETURN(QStringLiteral("terminal"));
}

QString FaceAuthKcm::retryAction() const
{
    switch (m_retryAction) {
    case RetryAction::None:
        return QStringLiteral("none");
    case RetryAction::Refresh:
        return QStringLiteral("refresh");
    case RetryAction::BeginEnrollment:
        return QStringLiteral("begin-enrollment");
    case RetryAction::CancelEnrollment:
        return QStringLiteral("cancel-enrollment");
    }
    Q_UNREACHABLE_RETURN(QStringLiteral("none"));
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
    if (busy()) {
        return;
    }
    const auto generation = ++m_generation;
    m_refreshRepliesPending = 2;
    m_refreshVersionSucceeded = false;
    m_refreshEnrollmentSucceeded = false;
    m_refreshVersionError.clear();
    m_refreshEnrollmentError.clear();
    setPageState(PageState::Loading);
    setRetryAction(RetryAction::None);
    setErrorCode({});
    setOutcome({});
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
            setSchemaVersion(0);
            finishRefreshReply(
                true, false, errorCodeFor(reply.error(), QStringLiteral("request-failed")));
            return;
        }
        setSchemaVersion(reply.value());
        if (reply.value() != supportedSchemaVersion) {
            finishRefreshReply(true, true, QStringLiteral("protocol-mismatch"));
            return;
        }
        finishRefreshReply(true, true);
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
            finishRefreshReply(
                false, false, errorCodeFor(reply.error(), QStringLiteral("request-failed")));
            return;
        }
        setEnrolled(reply.value());
        finishRefreshReply(false, true);
    });
}

void FaceAuthKcm::finishRefreshReply(bool versionReply,
                                     bool succeeded,
                                     const QString &errorCode)
{
    if (versionReply) {
        m_refreshVersionSucceeded = succeeded;
        m_refreshVersionError = errorCode;
    } else {
        m_refreshEnrollmentSucceeded = succeeded;
        m_refreshEnrollmentError = errorCode;
    }
    if (m_refreshRepliesPending == 0 || --m_refreshRepliesPending != 0) {
        return;
    }

    const bool available = m_refreshVersionSucceeded && m_refreshEnrollmentSucceeded;
    setDaemonAvailable(available);
    const QString failure = !m_refreshVersionError.isEmpty() ? m_refreshVersionError
                                                              : m_refreshEnrollmentError;
    if (available && failure.isEmpty() && m_schemaVersion == supportedSchemaVersion) {
        setErrorCode({});
        setRetryAction(RetryAction::None);
        setPageState(PageState::Ready);
        return;
    }

    setErrorCode(failure.isEmpty() ? QStringLiteral("request-failed") : failure);
    setRetryAction(failure == QLatin1String("protocol-mismatch") ? RetryAction::None
                                                                  : RetryAction::Refresh);
    setPageState(PageState::Terminal);
}

void FaceAuthKcm::beginEnrollment()
{
    if (busy() || !m_daemonAvailable || m_schemaVersion != supportedSchemaVersion) {
        return;
    }
    const auto generation = ++m_generation;
    setPageState(PageState::Authorizing);
    setRetryAction(RetryAction::None);
    setOperationId({});
    setCue({});
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
        if (reply.isError()) {
            bestEffortCancel();
            setOperationId({});
            setCue({});
            const QString errorCode =
                errorCodeFor(reply.error(), QStringLiteral("enrollment-failed"));
            setErrorCode(errorCode);
            setRetryAction(errorCode == QLatin1String("daemon-unavailable")
                               ? RetryAction::Refresh
                               : RetryAction::BeginEnrollment);
            setPageState(PageState::Terminal);
            return;
        }
        if (!isCanonicalOperationId(reply.value())
            || (!m_operationId.isEmpty() && m_operationId != reply.value())) {
            bestEffortCancel();
            setOperationId({});
            setCue({});
            setErrorCode(QStringLiteral("invalid-operation"));
            setRetryAction(RetryAction::Refresh);
            setPageState(PageState::Terminal);
            return;
        }
        setOperationId(reply.value());
        if (m_pageState == PageState::Authorizing) {
            setCue(QStringLiteral("preparing"));
            setPageState(PageState::Starting);
        }
    });
}

void FaceAuthKcm::cancelEnrollment()
{
    if (!canCancel()) {
        return;
    }
    const auto generation = m_generation;
    m_stateBeforeCancel = m_pageState;
    setPageState(PageState::Cancelling);
    setRetryAction(RetryAction::None);
    setErrorCode({});
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
            setPageState(m_stateBeforeCancel);
            setErrorCode(errorCodeFor(reply.error(), QStringLiteral("cancel-failed")));
            setRetryAction(RetryAction::CancelEnrollment);
        }
    });
}

void FaceAuthKcm::retry()
{
    const auto action = m_retryAction;
    setRetryAction(RetryAction::None);
    switch (action) {
    case RetryAction::None:
        return;
    case RetryAction::Refresh:
        refresh();
        return;
    case RetryAction::BeginEnrollment:
        beginEnrollment();
        return;
    case RetryAction::CancelEnrollment:
        cancelEnrollment();
        return;
    }
}

void FaceAuthKcm::handleServiceRegistered()
{
    refresh();
}

void FaceAuthKcm::handleServiceUnregistered()
{
    ++m_generation;
    m_refreshRepliesPending = 0;
    setDaemonAvailable(false);
    setSchemaVersion(0);
    setOperationId({});
    setCue({});
    setOutcome(QStringLiteral("unavailable"));
    setErrorCode(QStringLiteral("daemon-unavailable"));
    setRetryAction(RetryAction::Refresh);
    setPageState(PageState::Terminal);
}

void FaceAuthKcm::handleEnrollmentProgress(const QString &operationId, const QString &progress)
{
    if (!isCanonicalOperationId(operationId)) {
        if (busy()) {
            setErrorCode(QStringLiteral("protocol-mismatch"));
            setRetryAction(RetryAction::None);
        }
        return;
    }
    if (m_operationId.isEmpty() && m_pageState == PageState::Authorizing) {
        setOperationId(operationId);
    }
    if (operationId != m_operationId) {
        return;
    }
    if (!isKnownProgress(progress)) {
        setErrorCode(QStringLiteral("protocol-mismatch"));
        setRetryAction(RetryAction::None);
        return;
    }
    if (m_pageState == PageState::Cancelling) {
        return;
    }
    setCue(progress);
    setPageState(progress == QLatin1String("preparing") ? PageState::Starting
                                                         : PageState::Capturing);
}

void FaceAuthKcm::handleEnrollmentCompleted(const QString &operationId, const QString &result)
{
    if (!isCanonicalOperationId(operationId)) {
        if (busy()) {
            setErrorCode(QStringLiteral("protocol-mismatch"));
            setRetryAction(RetryAction::None);
        }
        return;
    }
    if (m_operationId.isEmpty() && m_pageState == PageState::Authorizing) {
        setOperationId(operationId);
    }
    if (operationId != m_operationId) {
        return;
    }
    ++m_generation;
    m_refreshRepliesPending = 0;
    setCue({});
    if (isKnownResult(result)) {
        setOutcome(experienceOutcome(result));
        setErrorCode({});
    } else {
        setOutcome(QStringLiteral("unavailable"));
        setErrorCode(QStringLiteral("protocol-mismatch"));
    }
    setOperationId({});
    setRetryAction(RetryAction::None);
    setPageState(PageState::Terminal);
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

void FaceAuthKcm::setPageState(PageState state)
{
    if (m_pageState == state) {
        return;
    }
    const bool wasBusy = busy();
    const bool wasCancellable = canCancel();
    m_pageState = state;
    Q_EMIT pageStateChanged();
    if (wasBusy != busy()) {
        Q_EMIT busyChanged();
    }
    if (wasCancellable != canCancel()) {
        Q_EMIT canCancelChanged();
    }
}

void FaceAuthKcm::setRetryAction(RetryAction action)
{
    if (m_retryAction == action) {
        return;
    }
    m_retryAction = action;
    Q_EMIT retryActionChanged();
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
    const bool wasCancellable = canCancel();
    m_operationId = operationId;
    if (wasBusy != busy()) {
        Q_EMIT busyChanged();
    }
    if (wasCancellable != canCancel()) {
        Q_EMIT canCancelChanged();
    }
}

void FaceAuthKcm::bestEffortCancel()
{
    if (m_manager == nullptr || m_operationId.isEmpty() || m_pageState == PageState::Cancelling) {
        return;
    }
    m_manager->call(QDBus::NoBlock,
                    QStringLiteral("CancelEnrollment"),
                    QVariant::fromValue(m_uid),
                    m_operationId);
}

#include "faceauthkcm.moc"
