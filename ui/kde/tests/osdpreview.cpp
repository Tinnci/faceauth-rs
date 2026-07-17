#include <KLocalizedContext>
#include <KLocalizedString>

#include <QGuiApplication>
#include <QImage>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QTimer>
#include <QUrl>

int main(int argc, char **argv)
{
    QGuiApplication application(argc, argv);
    if (QIcon::themeName().isEmpty()) {
        QIcon::setThemeName(QStringLiteral("breeze"));
    }
    if (application.arguments().size() < 2 || application.arguments().size() > 3) {
        return 2;
    }
    const QString output = application.arguments().value(2, QStringLiteral("/tmp/faceauth-osd-preview.png"));

    KLocalizedString::setApplicationDomain("faceauth");
    QQmlApplicationEngine engine;
    auto *translations = new KLocalizedContext(&engine);
    translations->setTranslationDomain(QStringLiteral("faceauth"));
    engine.rootContext()->setContextObject(translations);
    engine.load(QUrl::fromLocalFile(application.arguments().at(1)));
    if (engine.rootObjects().isEmpty()) {
        return 1;
    }
    auto *window = qobject_cast<QQuickWindow *>(engine.rootObjects().constFirst());
    if (window == nullptr) {
        return 1;
    }
    QTimer::singleShot(500, &application, [&application, output, window]() {
        const QImage image = window->grabWindow();
        application.exit(!image.isNull() && image.save(output) ? 0 : 1);
    });
    return application.exec();
}
