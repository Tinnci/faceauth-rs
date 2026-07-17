import QtQuick
import QtQuick.Controls as QQC2
import org.kde.kirigami as Kirigami

Item {
    id: root

    property string iconName: "edit-image-face-recognize"
    property bool running: false
    property string tone: "neutral"
    property string accessibleName: ""
    property bool reducedMotion: false
    readonly property color toneColor: {
        if (tone === "success")
            return Kirigami.Theme.positiveTextColor;

        if (tone === "error")
            return Kirigami.Theme.negativeTextColor;

        if (tone === "attention")
            return Kirigami.Theme.neutralTextColor;

        return Kirigami.Theme.highlightColor;
    }

    implicitWidth: Kirigami.Units.gridUnit * 8
    implicitHeight: implicitWidth
    Accessible.role: Accessible.Graphic
    Accessible.name: root.accessibleName

    Rectangle {
        id: pulse

        anchors.fill: parent
        radius: width / 2
        color: "transparent"
        border.color: root.toneColor
        border.width: 1
        opacity: 0
        scale: 0.86

        ParallelAnimation {
            running: root.running && !root.reducedMotion
            loops: Animation.Infinite

            NumberAnimation {
                target: pulse
                property: "scale"
                from: 0.86
                to: 1.16
                duration: 1450
                easing.type: Easing.OutCubic
            }

            SequentialAnimation {
                NumberAnimation {
                    target: pulse
                    property: "opacity"
                    from: 0
                    to: 0.34
                    duration: 420
                    easing.type: Easing.OutCubic
                }

                NumberAnimation {
                    target: pulse
                    property: "opacity"
                    from: 0.34
                    to: 0
                    duration: 1030
                    easing.type: Easing.InCubic
                }
            }
        }
    }

    Rectangle {
        id: surface

        anchors.fill: parent
        radius: width / 2
        color: Kirigami.ColorUtils.tintWithAlpha(Kirigami.Theme.backgroundColor, root.toneColor, 0.12)
        border.color: root.toneColor
        border.width: Math.max(2, Kirigami.Units.smallSpacing / 2)
        scale: root.tone === "success" ? 1.04 : 1

        Kirigami.Icon {
            anchors.centerIn: parent
            width: parent.width * 0.46
            height: width
            visible: root.tone === "neutral"
            source: root.iconName
            scale: root.running ? 0.94 : 1

            Behavior on scale {
                NumberAnimation {
                    duration: root.reducedMotion ? 0 : 180
                    easing.type: Easing.OutCubic
                }
            }
        }

        QQC2.Label {
            anchors.centerIn: parent
            visible: root.tone !== "neutral"
            text: root.tone === "success" ? "✓" : "!"
            color: root.toneColor
            font.pixelSize: parent.width * 0.42
            font.bold: true
            opacity: root.tone === "neutral" ? 0 : 1

            Behavior on opacity {
                NumberAnimation {
                    duration: root.reducedMotion ? 0 : 150
                }
            }
        }

        QQC2.BusyIndicator {
            anchors.fill: parent
            anchors.margins: Kirigami.Units.smallSpacing
            running: root.running
            visible: running
        }

        Behavior on color {
            ColorAnimation {
                duration: root.reducedMotion ? 0 : 180
            }
        }

        Behavior on border.color {
            ColorAnimation {
                duration: root.reducedMotion ? 0 : 180
            }
        }

        Behavior on scale {
            NumberAnimation {
                duration: root.reducedMotion ? 0 : 260
                easing.type: Easing.OutBack
            }
        }
    }
}
