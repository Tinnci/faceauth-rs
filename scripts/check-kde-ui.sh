#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

find_qt_tool() {
    local name=$1
    if [[ -x "/usr/lib/qt6/bin/$name" ]]; then
        printf '%s\n' "/usr/lib/qt6/bin/$name"
    elif command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
    else
        printf 'required Qt tool is unavailable: %s\n' "$name" >&2
        return 1
    fi
}

qmlformat=$(find_qt_tool qmlformat)
qmllint=$(find_qt_tool qmllint)
kcmshell=$(command -v kcmshell6)
timeout_command=$(command -v timeout)
qml_files=(
    ui/kde/ui/main.qml
    ui/kde/ui/components/FaceStatusRing.qml
    ui/kde/ui/components/GuidancePanel.qml
    ui/kde/ui/osd/FaceAuthOsd.qml
    ui/kde/tests/OsdPreview.qml
)

temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT

for file in "${qml_files[@]}"; do
    formatted="$temporary/$(basename "$file")"
    "$qmlformat" "$file" >"$formatted"
    if ! cmp --silent "$file" "$formatted"; then
        diff --unified "$file" "$formatted" || true
        printf 'QML formatting differs: %s\n' "$file" >&2
        exit 1
    fi
done

for state in loading ready authorizing starting capturing cancelling terminal; do
    if ! grep -Fq "QStringLiteral(\"$state\")" ui/kde/faceauthkcm.cpp; then
        printf 'KCM page state is missing from the adapter: %s\n' "$state" >&2
        exit 1
    fi
done

for action in none refresh begin-enrollment cancel-enrollment; do
    if ! grep -Fq "QStringLiteral(\"$action\")" ui/kde/faceauthkcm.cpp; then
        printf 'KCM retry action is missing from the adapter: %s\n' "$action" >&2
        exit 1
    fi
done

grep -Fq 'QDBus::NoBlock' ui/kde/faceauthkcm.cpp
grep -Fq 'QDBusConnection::connectToBus(QDBusConnection::SystemBus' ui/kde/faceauthkcm.cpp
grep -Fq 'QDBusConnection::disconnectFromBus(m_busConnectionName)' ui/kde/faceauthkcm.cpp
grep -Fq 'if (!canCancel())' ui/kde/faceauthkcm.cpp
grep -Fq 'setPageState(PageState::Cancelling)' ui/kde/faceauthkcm.cpp
grep -Fq 'root.backend.retry()' ui/kde/ui/main.qml
grep -Fq 'root.backend.canCancel || root.cancelling' ui/kde/ui/main.qml
grep -Fq 'i18n("Cancelling…")' ui/kde/ui/main.qml
grep -Fq '"authorizing": i18n("Authorize face setup")' ui/kde/ui/components/GuidancePanel.qml
grep -Fq '"cancelling": i18n("Cancelling enrollment…")' ui/kde/ui/components/GuidancePanel.qml
grep -Fq '"X-KDE-System-Settings-Parent-Category": "security-privacy"' ui/kde/kcm_faceauth.json

"$qmllint" -I ui/kde/ui -I /usr/lib/qt6/qml "${qml_files[@]}"

cmake -S ui/kde -B "$temporary/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX="$temporary/install" \
    -DFACEAUTH_BUILD_PREVIEW=ON
cmake --build "$temporary/build" --parallel 2
cmake --install "$temporary/build"

qt_plugin_path="$temporary/install/lib/plugins"
if [[ -n "${QT_PLUGIN_PATH:-}" ]]; then
    qt_plugin_path="$qt_plugin_path:$QT_PLUGIN_PATH"
fi
xdg_data_dirs="$temporary/install/share"
if [[ -n "${XDG_DATA_DIRS:-}" ]]; then
    xdg_data_dirs="$xdg_data_dirs:$XDG_DATA_DIRS"
else
    xdg_data_dirs="$xdg_data_dirs:/usr/local/share:/usr/share"
fi
mkdir -p "$temporary/cache" "$temporary/config" "$temporary/data" "$temporary/runtime"
chmod 700 "$temporary/runtime"
kde_test_environment=(
    "QT_PLUGIN_PATH=$qt_plugin_path"
    "XDG_CACHE_HOME=$temporary/cache"
    "XDG_CONFIG_HOME=$temporary/config"
    "XDG_DATA_HOME=$temporary/data"
    "XDG_DATA_DIRS=$xdg_data_dirs"
    "XDG_RUNTIME_DIR=$temporary/runtime"
    "QT_QPA_PLATFORM=offscreen"
    "QT_QUICK_BACKEND=software"
)
module_list=$("$timeout_command" 20s env "${kde_test_environment[@]}" "$kcmshell" --list)
if ! grep -Fq 'kcm_faceauth' <<<"$module_list"; then
    printf 'installed KCM is not discoverable through kcmshell6 --list\n' >&2
    exit 1
fi
"$timeout_command" 20s env "${kde_test_environment[@]}" "$kcmshell" --smoke-test kcm_faceauth

QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software \
    "$temporary/build/bin/faceauth-osd-preview" \
    "$root/ui/kde/tests/OsdPreview.qml" "$temporary/faceauth-osd-preview.png"
test -s "$temporary/faceauth-osd-preview.png"
test -s "$temporary/install/lib/plugins/plasma/kcms/systemsettings/kcm_faceauth.so"
