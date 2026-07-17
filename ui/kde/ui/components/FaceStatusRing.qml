import QtQuick
import QtQuick.Controls as QQC2
import org.kde.kirigami as Kirigami

Item {
    id: root

    property string iconName: "edit-image-face-recognize"
    property bool running: false
    property string tone: "neutral"
    property string accessibleName: ""
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
        anchors.fill: parent
        radius: width / 2
        color: Kirigami.ColorUtils.tintWithAlpha(Kirigami.Theme.backgroundColor, root.toneColor, 0.12)
        border.color: root.toneColor
        border.width: Math.max(2, Kirigami.Units.smallSpacing / 2)

        Kirigami.Icon {
            anchors.centerIn: parent
            width: parent.width * 0.46
            height: width
            visible: root.tone === "neutral"
            source: root.iconName
        }

        QQC2.Label {
            anchors.centerIn: parent
            visible: root.tone !== "neutral"
            text: root.tone === "success" ? "✓" : "!"
            color: root.toneColor
            font.pixelSize: parent.width * 0.42
            font.bold: true
        }

        QQC2.BusyIndicator {
            anchors.fill: parent
            anchors.margins: Kirigami.Units.smallSpacing
            running: root.running
            visible: running
        }
    }
}
