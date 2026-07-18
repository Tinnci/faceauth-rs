#pragma once

#include <KQuickConfigModule>

class QDBusInterface;
class QDBusServiceWatcher;

class FaceAuthKcm final : public KQuickConfigModule
{
    Q_OBJECT
    Q_PROPERTY(bool daemonAvailable READ daemonAvailable NOTIFY daemonAvailableChanged)
    Q_PROPERTY(bool enrolled READ enrolled NOTIFY enrolledChanged)
    Q_PROPERTY(bool loading READ loading NOTIFY pageStateChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    Q_PROPERTY(bool canCancel READ canCancel NOTIFY canCancelChanged)
    Q_PROPERTY(QString pageState READ pageState NOTIFY pageStateChanged)
    Q_PROPERTY(QString retryAction READ retryAction NOTIFY retryActionChanged)
    Q_PROPERTY(quint16 schemaVersion READ schemaVersion NOTIFY schemaVersionChanged)
    Q_PROPERTY(QString cue READ cue NOTIFY cueChanged)
    Q_PROPERTY(QString outcome READ outcome NOTIFY outcomeChanged)
    Q_PROPERTY(QString errorCode READ errorCode NOTIFY errorCodeChanged)

public:
    explicit FaceAuthKcm(QObject *parent, const KPluginMetaData &metaData);
    ~FaceAuthKcm() override;

    [[nodiscard]] bool daemonAvailable() const;
    [[nodiscard]] bool enrolled() const;
    [[nodiscard]] bool loading() const;
    [[nodiscard]] bool busy() const;
    [[nodiscard]] bool canCancel() const;
    [[nodiscard]] QString pageState() const;
    [[nodiscard]] QString retryAction() const;
    [[nodiscard]] quint16 schemaVersion() const;
    [[nodiscard]] QString cue() const;
    [[nodiscard]] QString outcome() const;
    [[nodiscard]] QString errorCode() const;

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void beginEnrollment();
    Q_INVOKABLE void cancelEnrollment();
    Q_INVOKABLE void retry();

Q_SIGNALS:
    void daemonAvailableChanged();
    void enrolledChanged();
    void pageStateChanged();
    void busyChanged();
    void canCancelChanged();
    void retryActionChanged();
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
    enum class PageState {
        Loading,
        Ready,
        Authorizing,
        Starting,
        Capturing,
        Cancelling,
        Terminal,
    };

    enum class RetryAction {
        None,
        Refresh,
        BeginEnrollment,
        CancelEnrollment,
    };

    void refreshVersion(quint64 generation);
    void refreshEnrollmentState(quint64 generation);
    void finishRefreshReply(bool versionReply, bool succeeded, const QString &errorCode = {});
    void bestEffortCancel();
    void setDaemonAvailable(bool available);
    void setEnrolled(bool enrolled);
    void setPageState(PageState state);
    void setRetryAction(RetryAction action);
    void setSchemaVersion(quint16 version);
    void setCue(const QString &cue);
    void setOutcome(const QString &outcome);
    void setErrorCode(const QString &errorCode);
    void setOperationId(const QString &operationId);
    QString m_busConnectionName;
    QDBusInterface *m_manager = nullptr;
    QDBusServiceWatcher *m_serviceWatcher = nullptr;
    quint32 m_uid = 0;
    quint64 m_generation = 0;
    quint16 m_schemaVersion = 0;
    quint8 m_refreshRepliesPending = 0;
    bool m_daemonAvailable = false;
    bool m_enrolled = false;
    bool m_refreshVersionSucceeded = false;
    bool m_refreshEnrollmentSucceeded = false;
    PageState m_pageState = PageState::Loading;
    PageState m_stateBeforeCancel = PageState::Capturing;
    RetryAction m_retryAction = RetryAction::None;
    QString m_refreshVersionError;
    QString m_refreshEnrollmentError;
    QString m_cue;
    QString m_outcome;
    QString m_errorCode;
    QString m_operationId;
};
