pragma ComponentBehavior: Bound
// The secure OSD host injects the KI18n context object.
// qmllint disable unqualified

import "../components"
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Rectangle {
    id: root

    property string cue: ""
    property string outcome: ""
    property bool running: cue.length > 0 && outcome.length === 0
    property bool passwordFallbackVisible: true

    implicitWidth: Kirigami.Units.gridUnit * 20
    implicitHeight: content.implicitHeight + Kirigami.Units.largeSpacing * 2
    radius: Kirigami.Units.cornerRadius
    color: Kirigami.Theme.backgroundColor
    border.color: Kirigami.Theme.disabledTextColor
    border.width: 1
    Accessible.role: Accessible.AlertMessage
    Accessible.name: i18n("Face authentication")

    RowLayout {
        id: content

        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing
        spacing: Kirigami.Units.largeSpacing

        FaceStatusRing {
            Layout.preferredWidth: Kirigami.Units.gridUnit * 5
            Layout.preferredHeight: Layout.preferredWidth
            running: root.running
            iconName: root.outcome === "succeeded" ? "dialog-ok-apply" : "edit-image-face-recognize"
            tone: root.outcome === "succeeded" ? "success" : (root.outcome === "try-again" || root.outcome === "unavailable" ? "error" : (root.outcome === "timed-out" ? "attention" : "neutral"))
            accessibleName: i18n("Face authentication status")
        }

        ColumnLayout {
            Layout.fillWidth: true

            GuidancePanel {
                Layout.fillWidth: true
                cue: root.cue
                outcome: root.outcome
                running: root.running
                mode: "authentication"
            }

            QQC2.Label {
                Layout.fillWidth: true
                visible: root.passwordFallbackVisible
                text: i18n("You can use your password instead")
                color: Kirigami.Theme.textColor
                opacity: 0.72
                wrapMode: Text.Wrap
            }
        }
    }
}
