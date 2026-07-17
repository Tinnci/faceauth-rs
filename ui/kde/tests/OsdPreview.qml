import "../ui/osd"
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window

    width: 520
    height: previews.implicitHeight + 32
    visible: true
    color: "#20242b"

    ColumnLayout {
        id: previews

        anchors.centerIn: parent
        spacing: 12

        FaceAuthOsd {
            Layout.preferredWidth: 440
            cue: "blink"
            passwordFallbackVisible: true
        }

        FaceAuthOsd {
            Layout.preferredWidth: 440
            outcome: "succeeded"
            passwordFallbackVisible: false
        }

        FaceAuthOsd {
            Layout.preferredWidth: 440
            outcome: "try-again"
            passwordFallbackVisible: true
        }
    }
}
