pragma ComponentBehavior: Bound
// The KCM or secure OSD host injects the KI18n context object.
// qmllint disable unqualified

import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

ColumnLayout {
    id: root

    property string cue: ""
    property string outcome: ""
    property bool running: false
    property string mode: "authentication"
    property bool reducedMotion: false

    readonly property string guidanceIcon: {
        if (root.outcome === "succeeded")
            return "dialog-ok-apply";
        if (root.outcome === "timed-out")
            return "chronometer";
        if (root.outcome === "cancelled")
            return "dialog-cancel";
        if (root.outcome.length > 0)
            return "data-warning";
        const icons = {
            "preparing": "view-refresh",
            "position-face": "edit-image-face-recognize",
            "hold-still": "media-playback-pause",
            "active-challenge": "system-run",
            "blink": "face-smile",
            "turn-left": "go-previous",
            "turn-right": "go-next",
            "return-to-center": "go-home",
            "processing": "document-encrypt"
        };
        return icons[root.cue] || "edit-image-face-recognize";
    }

    function titleForCue(value) {
        const titles = {
            "authorizing": i18n("Authorize face setup"),
            "cancelling": i18n("Cancelling enrollment…"),
            "preparing": i18n("Getting the cameras ready…"),
            "position-face": i18n("Center your face"),
            "hold-still": i18n("Hold still"),
            "active-challenge": i18n("Follow the on-screen action"),
            "blink": i18n("Blink now"),
            "turn-left": i18n("Turn your head left"),
            "turn-right": i18n("Turn your head right"),
            "return-to-center": i18n("Look straight ahead"),
            "processing": i18n("Checking securely…")
        };
        return titles[value] || "";
    }

    function bodyForCue(value) {
        const bodies = {
            "authorizing": i18n("Complete or dismiss the administrator authorization prompt."),
            "cancelling": i18n("Waiting for the service to stop camera and enrollment work."),
            "preparing": i18n("This should only take a moment."),
            "position-face": i18n("Keep your face inside the guide and look at the camera."),
            "hold-still": i18n("Keep your eyes open and maintain a natural expression."),
            "active-challenge": i18n("The action is randomized to verify that you are present."),
            "blink": i18n("Close both eyes once, then open them."),
            "turn-left": i18n("Turn naturally, without moving out of the guide."),
            "turn-right": i18n("Turn naturally, without moving out of the guide."),
            "return-to-center": i18n("Return to a centered, eyes-open position."),
            "processing": i18n("You can remain still while encrypted evidence is prepared.")
        };
        return bodies[value] || "";
    }

    function titleForOutcome(value) {
        const authenticationTitles = {
            "succeeded": i18n("Face recognized"),
            "try-again": i18n("Try face authentication again"),
            "cancelled": i18n("Face authentication cancelled"),
            "timed-out": i18n("Face authentication timed out"),
            "unavailable": i18n("Face authentication is unavailable")
        };
        const enrollmentTitles = {
            "succeeded": i18n("Face authentication is ready"),
            "cancelled": i18n("Enrollment cancelled"),
            "timed-out": i18n("Enrollment timed out"),
            "unavailable": i18n("Face authentication is unavailable")
        };
        const titles = root.mode === "enrollment" ? enrollmentTitles : authenticationTitles;
        return titles[value] || "";
    }

    spacing: Kirigami.Units.smallSpacing

    Kirigami.Icon {
        Layout.alignment: Qt.AlignHCenter
        Layout.preferredWidth: Kirigami.Units.iconSizes.medium
        Layout.preferredHeight: Layout.preferredWidth
        source: root.guidanceIcon
        opacity: root.cue.length > 0 || root.outcome.length > 0 ? 1 : 0
        scale: root.running ? 0.94 : 1

        Behavior on opacity {
            NumberAnimation {
                duration: root.reducedMotion ? 0 : 140
            }
        }

        Behavior on scale {
            NumberAnimation {
                duration: root.reducedMotion ? 0 : 180
                easing.type: Easing.OutCubic
            }
        }
    }

    QQC2.Label {
        Layout.fillWidth: true
        text: root.outcome.length > 0 ? root.titleForOutcome(root.outcome) : root.titleForCue(root.cue)
        font.pointSize: Kirigami.Theme.defaultFont.pointSize * 1.15
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.Wrap
    }

    QQC2.Label {
        Layout.fillWidth: true
        visible: text.length > 0
        text: root.outcome.length > 0 ? "" : root.bodyForCue(root.cue)
        color: Kirigami.Theme.textColor
        opacity: 0.72
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.Wrap
    }
}
