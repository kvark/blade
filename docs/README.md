# Blade

![](logo.png)

Blade is a low-level GPU library in Rust with the focus on ergonomics.
In other words, Blade is an attempt to make graphics programming fun with Rust.
It doesn't try to be comprehensive or safe, it trusts you.

See [motivation](motivation.md), [FAQ](FAQ.md), and [performance](performance.md) for more info.

## Platforms

The backend is selected automatically based on the host platform.

Vulkan:
- Desktop Linux/Windows
- Android

Metal:
- Desktop macOS
- iOS

## Instructions

Check:
```
cargo check
```
Run the minimal example:
```
cargo run --example mini
```

### Vulkan Portability

First, ensure to load the environment from the Vulkan SDK:
```bash
cd /opt/VulkanSDK && source setup-env.sh
```

Vulkan backend can be forced on using "portability" config flag. Example invocation that produces a portability build into another target folder:
```bash
RUSTFLAGS="--cfg portability" CARGO_TARGET_DIR=./target-vk cargo test
```
