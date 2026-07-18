pragma ComponentBehavior: Bound
// The KCM host injects both `kcm` and the KI18n context object.
// qmllint disable unqualified

import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import "components"
import org.kde.kcmutils as KCMUtils
import org.kde.kirigami as Kirigami

Kirigami.ScrollablePage {
    id: root

    readonly property var backend: kcm
    readonly property bool cancelling: root.backend.pageState === "cancelling"

    function accessibleStatus() {
        if (root.backend.loading)
            return i18n("Checking face authentication status");

        if (root.backend.pageState === "authorizing")
            return i18n("Waiting for enrollment authorization");

        if (root.backend.busy)
            return i18n("Face enrollment in progress");

        if (root.backend.outcome === "succeeded")
            return i18n("Face enrollment completed");

        if (root.backend.outcome === "unavailable")
            return i18n("Face enrollment unavailable");

        return root.backend.enrolled ? i18n("Face authentication enrolled") : i18n("Face authentication not enrolled");
    }

    function errorText(code) {
        const messages = {
            "daemon-unavailable": i18n("The face authentication service is not available."),
            "authorization-failed": i18n("Administrator authorization was not granted."),
            "protocol-mismatch": i18n("The service and settings module use incompatible versions."),
            "invalid-operation": i18n("The service returned an invalid enrollment operation."),
            "cancel-failed": i18n("The enrollment operation could not be cancelled."),
            "enrollment-failed": i18n("Enrollment could not be started."),
            "request-failed": i18n("The service did not complete the request.")
        };
        return messages[code] || "";
    }

    function headlineText() {
        if (root.backend.loading)
            return i18n("Checking face authentication…");

        return root.backend.enrolled ? i18n("Your face is enrolled") : i18n("Set up face authentication");
    }

    function retryText(action) {
        if (action === "begin-enrollment")
            return i18n("Try Enrollment Again");

        if (action === "cancel-enrollment")
            return i18n("Try Cancelling Again");

        return i18n("Reconnect");
    }

    function statusText() {
        if (root.backend.loading)
            return i18n("Reading your enrollment state securely.");

        if (root.backend.enrolled)
            return i18n("Use the infrared and visible cameras for supported unlock and authorization prompts.");

        return i18n("Register with the infrared and visible cameras. Your password will remain available.");
    }

    title: i18n("Face Authentication")
    implicitWidth: Kirigami.Units.gridUnit * 34
    implicitHeight: Kirigami.Units.gridUnit * 30
    KCMUtils.ConfigModule.buttons: KCMUtils.ConfigModule.Help

    ColumnLayout {
        spacing: Kirigami.Units.largeSpacing

        Kirigami.InlineMessage {
            Layout.fillWidth: true
            visible: root.backend.errorCode.length > 0
            type: Kirigami.MessageType.Error
            text: root.errorText(root.backend.errorCode)
            actions: [
                Kirigami.Action {
                    visible: root.backend.retryAction !== "none"
                    text: root.retryText(root.backend.retryAction)
                    icon.name: "view-refresh"
                    onTriggered: root.backend.retry()
                }
            ]
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: Kirigami.Units.largeSpacing

            FaceStatusRing {
                Layout.alignment: Qt.AlignHCenter
                running: root.backend.loading || root.backend.busy
                iconName: root.backend.outcome === "succeeded" ? "dialog-ok-apply" : (root.backend.outcome === "timed-out" ? "chronometer" : (root.backend.outcome === "cancelled" ? "dialog-cancel" : "edit-image-face-recognize"))
                tone: root.backend.outcome === "succeeded" ? "success" : (root.backend.outcome === "unavailable" ? "error" : (root.backend.outcome === "timed-out" ? "attention" : "neutral"))
                accessibleName: root.accessibleStatus()
            }

            QQC2.Label {
                Layout.fillWidth: true
                text: root.headlineText()
                font.pointSize: Kirigami.Theme.defaultFont.pointSize * 1.35
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }

            QQC2.Label {
                Layout.fillWidth: true
                text: root.statusText()
                color: Kirigami.Theme.textColor
                opacity: 0.72
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.Wrap
            }

            GuidancePanel {
                Layout.fillWidth: true
                visible: root.backend.busy || root.backend.outcome.length > 0
                cue: root.backend.pageState === "authorizing" ? "authorizing" : (root.cancelling ? "cancelling" : root.backend.cue)
                outcome: root.backend.outcome
                running: root.backend.busy
                mode: "enrollment"
            }

            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                spacing: Kirigami.Units.smallSpacing

                QQC2.Button {
                    text: root.backend.enrolled ? i18n("Update Face") : i18n("Set Up")
                    icon.name: "camera-web"
                    enabled: !root.backend.loading && root.backend.daemonAvailable && root.backend.schemaVersion === 2 && !root.backend.busy
                    onClicked: root.backend.beginEnrollment()
                }

                QQC2.Button {
                    visible: root.backend.canCancel || root.cancelling
                    text: root.cancelling ? i18n("Cancelling…") : i18n("Cancel")
                    icon.name: root.cancelling ? "view-refresh" : "dialog-cancel"
                    enabled: root.backend.canCancel
                    onClicked: root.backend.cancelEnrollment()
                }
            }
        }

        Kirigami.Separator {
            Layout.fillWidth: true
        }

        Kirigami.AbstractCard {
            Layout.fillWidth: true

            contentItem: RowLayout {
                spacing: Kirigami.Units.largeSpacing

                Kirigami.Icon {
                    source: "security-high"
                    Layout.preferredWidth: Kirigami.Units.iconSizes.medium
                    Layout.preferredHeight: Layout.preferredWidth
                }

                ColumnLayout {
                    Layout.fillWidth: true

                    QQC2.Label {
                        Layout.fillWidth: true
                        text: i18n("Private by design")
                        font.bold: true
                    }

                    QQC2.Label {
                        Layout.fillWidth: true
                        text: i18n("Raw camera images are not retained. Only an encrypted derived face template is stored, and password fallback stays enabled.")
                        wrapMode: Text.Wrap
                    }
                }
            }
        }

        QQC2.Label {
            Layout.fillWidth: true
            text: root.backend.loading ? i18n("Checking service…") : (root.backend.daemonAvailable ? i18n("Service connected · management schema %1", root.backend.schemaVersion) : i18n("Service disconnected"))
            color: Kirigami.Theme.textColor
            opacity: 0.72
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
