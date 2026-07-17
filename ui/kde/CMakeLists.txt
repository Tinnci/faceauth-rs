cmake_minimum_required(VERSION 3.25)

project(faceauth-kde VERSION 0.1.0 LANGUAGES CXX)

option(FACEAUTH_BUILD_PREVIEW "Build the offscreen OSD visual-test host" OFF)

set(QT_MIN_VERSION "6.6.0")
set(KF_MIN_VERSION "6.0.0")

find_package(ECM ${KF_MIN_VERSION} REQUIRED NO_MODULE)
list(APPEND CMAKE_MODULE_PATH "${ECM_MODULE_PATH}")

include(KDEInstallDirs)
include(KDECMakeSettings)
include(KDECompilerSettings NO_POLICY_SCOPE)

find_package(Qt6 ${QT_MIN_VERSION} REQUIRED COMPONENTS Core DBus Qml Quick)
find_package(KF6 ${KF_MIN_VERSION} REQUIRED COMPONENTS CoreAddons I18n KCMUtils Kirigami)

kcmutils_add_qml_kcm(kcm_faceauth
    SOURCES
        faceauthkcm.cpp
        faceauthkcm.h
)

target_compile_features(kcm_faceauth PRIVATE cxx_std_20)
target_link_libraries(kcm_faceauth PRIVATE
    Qt6::Core
    Qt6::DBus
    KF6::CoreAddons
    KF6::I18n
    KF6::KCMUtilsQuick
)

if(FACEAUTH_BUILD_PREVIEW)
    add_executable(faceauth-osd-preview tests/osdpreview.cpp)
    target_compile_features(faceauth-osd-preview PRIVATE cxx_std_20)
    target_link_libraries(faceauth-osd-preview PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Qml
        Qt6::Quick
        KF6::I18n
    )
endif()
