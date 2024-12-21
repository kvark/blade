### Testing VR example (Android)

This crate builds the OpenXR sample app (`xr`) for Android.

Prerequisites:
- Android SDK + NDK configured (`ANDROID_HOME`/`ANDROID_NDK_HOME`)
- `adb` available in `PATH`
- `cargo-apk` installed (`cargo install cargo-apk`)
- Rust Android target installed (`rustup target add aarch64-linux-android`)
- A connected OpenXR-capable Android HMD (for example Meta Quest) with developer mode enabled

Build and run:
```bash
cargo apk run --manifest-path examples-android/Cargo.toml --no-logcat
```

Build:
```bash
cargo apk build --manifest-path examples-android/Cargo.toml
adb install -r target\debug\apk\xr.apk
```
`--release` builds require configuring a release keystore under `[package.metadata.android.signing.release]`.

Run:
```bash
adb shell am force-stop rust.xr
adb logcat -c
adb shell am start -n rust.xr/android.app.NativeActivity
```

Getting the logs:
```bash
adb shell pidof rust.xr
adb logcat -d --pid <pid>
```

Do everything in one command line:
```bash
cargo apk build --manifest-path examples-android/Cargo.toml --release && adb shell am force-stop rust.xr && adb install -r target/release/apk/xr.apk && adb shell am start -n rust.xr/android.app.NativeActivity &&  adb logcat -v time | grep -E "blade-xr|RustStdoutStderr|XR mark:"
```
