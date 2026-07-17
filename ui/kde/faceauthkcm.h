#pragma once

#include <KQuickConfigModule>

class QDBusInterface;
class QDBusServiceWatcher;

class FaceAuthKcm final : public KQuickConfigModule
{
    Q_OBJECT
    Q_PROPERTY(bool daemonAvailable READ daemonAvailable NOTIFY daemonAvailableChanged)
    Q_PROPERTY(bool enrolled READ enrolled NOTIFY enrolledChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(quint16 schemaVersion READ schemaVersion NOTIFY schemaVersionChanged)
    Q_PROPERTY(QString cue READ cue NOTIFY cueChanged)
    Q_PROPERTY(QString outcome READ outcome NOTIFY outcomeChanged)
    Q_PROPERTY(QString errorCode READ errorCode NOTIFY errorCodeChanged)

public:
    explicit FaceAuthKcm(QObject *parent, const KPluginMetaData &metaData);
    ~FaceAuthKcm() override;

    [[nodiscard]] bool daemonAvailable() const;
    [[nodiscard]] bool enrolled() const;
    [[nodiscard]] bool busy() const;
    [[nodiscard]] quint16 schemaVersion() const;
    [[nodiscard]] QString cue() const;
    [[nodiscard]] QString outcome() const;
    [[nodiscard]] QString errorCode() const;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void beginEnrollment();
    Q_INVOKABLE void cancelEnrollment();

Q_SIGNALS:
    void daemonAvailableChanged();
    void enrolledChanged();
    void busyChanged();
    void schemaVersionChanged();
    void cueChanged();
    void outcomeChanged();
    void errorCodeChanged();

private Q_SLOTS:
    void handleServiceRegistered();
    void handleServiceUnregistered();
    void handleEnrollmentProgress(const QString &operationId, const QString &progress);
    void handleEnrollmentCompleted(const QString &operationId, const QString &result);

private:
    void refreshVersion(quint64 generation);
    void refreshEnrollmentState(quint64 generation);
    void setDaemonAvailable(bool available);
    void setEnrolled(bool enrolled);
    void setStarting(bool starting);
    void setSchemaVersion(quint16 version);
    void setCue(const QString &cue);
    void setOutcome(const QString &outcome);
    void setErrorCode(const QString &errorCode);
    void setOperationId(const QString &operationId);
    QDBusInterface *m_manager = nullptr;
    QDBusServiceWatcher *m_serviceWatcher = nullptr;
    quint32 m_uid = 0;
    quint64 m_generation = 0;
    quint16 m_schemaVersion = 0;
    bool m_daemonAvailable = false;
    bool m_enrolled = false;
    bool m_starting = false;
    QString m_cue;
    QString m_outcome;
    QString m_errorCode;
    QString m_operationId;
};
