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

"$qmllint" -I ui/kde/ui -I /usr/lib/qt6/qml "${qml_files[@]}"

cmake -S ui/kde -B "$temporary/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX="$temporary/install" \
    -DFACEAUTH_BUILD_PREVIEW=ON
cmake --build "$temporary/build" --parallel 2
cmake --install "$temporary/build"

QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software \
    "$temporary/build/bin/faceauth-osd-preview" \
    "$root/ui/kde/tests/OsdPreview.qml" "$temporary/faceauth-osd-preview.png"
test -s "$temporary/faceauth-osd-preview.png"
test -s "$temporary/install/lib/plugins/plasma/kcms/systemsettings/kcm_faceauth.so"
