#!/usr/bin/env swift

import Foundation
import Metal

private let shaderSource = """
  #include <metal_stdlib>
  using namespace metal;

  kernel void bump(device uint &value [[buffer(0)]]) {
      value += 1;
  }
  """

private enum Workload: String {
  case independent
  case dependent
}

private enum Tracking: String {
  case tracked
  case untracked
  case untrackedFenced = "untracked+fence"
  case untrackedEvent = "untracked+event"
}

private struct Measurement {
  var encodeMicroseconds: Double
  var commitMicroseconds: Double
  var gpuMicroseconds: Double
  var wallMicroseconds: Double
}

private struct Stats {
  var median: Double
  var p05: Double
  var p95: Double
}

private func stats(_ input: [Double]) -> Stats {
  let values = input.sorted()
  func percentile(_ p: Double) -> Double {
    let index = Int((Double(values.count - 1) * p).rounded())
    return values[index]
  }
  return Stats(
    median: percentile(0.5),
    p05: percentile(0.05),
    p95: percentile(0.95)
  )
}

private func resourceOptions(_ tracking: Tracking) -> MTLResourceOptions {
  var options: MTLResourceOptions = [.storageModeShared]
  switch tracking {
  case .tracked:
    options.insert(.hazardTrackingModeTracked)
  case .untracked, .untrackedFenced, .untrackedEvent:
    options.insert(.hazardTrackingModeUntracked)
  }
  return options
}

private final class Benchmark {
  let device: MTLDevice
  let queue: MTLCommandQueue
  let pipeline: MTLComputePipelineState

  init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw NSError(domain: "MetalHazardBench", code: 1)
    }
    guard let queue = device.makeCommandQueue() else {
      throw NSError(domain: "MetalHazardBench", code: 2)
    }
    let library = try device.makeLibrary(source: shaderSource, options: nil)
    guard let function = library.makeFunction(name: "bump") else {
      throw NSError(domain: "MetalHazardBench", code: 3)
    }
    self.device = device
    self.queue = queue
    self.pipeline = try device.makeComputePipelineState(function: function)
  }

  func makeCase(
    workload: Workload,
    tracking: Tracking,
    passCount: Int
  ) -> BenchmarkCase {
    BenchmarkCase(
      benchmark: self,
      workload: workload,
      tracking: tracking,
      passCount: passCount
    )
  }
}

private final class BenchmarkCase {
  let benchmark: Benchmark
  let workload: Workload
  let tracking: Tracking
  let passCount: Int
  let buffers: [MTLBuffer]
  let fence: MTLFence?
  let event: MTLEvent?
  var nextEventValue: UInt64 = 1

  init(
    benchmark: Benchmark,
    workload: Workload,
    tracking: Tracking,
    passCount: Int
  ) {
    precondition(
      workload == .dependent
        || (tracking != .untrackedFenced && tracking != .untrackedEvent)
    )
    self.benchmark = benchmark
    self.workload = workload
    self.tracking = tracking
    self.passCount = passCount
    let bufferCount = workload == .independent ? passCount : 1
    self.buffers = (0..<bufferCount).map { _ in
      benchmark.device.makeBuffer(
        length: MemoryLayout<UInt32>.stride,
        options: resourceOptions(tracking)
      )!
    }
    self.fence =
      tracking == .untrackedFenced
      ? benchmark.device.makeFence()
      : nil
    self.event =
      tracking == .untrackedEvent
      ? benchmark.device.makeEvent()
      : nil
  }

  func run() -> (Measurement, Bool) {
    for buffer in buffers {
      buffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    }

    let wallStart = ContinuousClock.now
    let encodeStart = ContinuousClock.now
    let commandBuffer = benchmark.queue.makeCommandBuffer()!
    for pass in 0..<passCount {
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      if let fence, pass != 0 {
        encoder.waitForFence(fence)
      }
      encoder.setComputePipelineState(benchmark.pipeline)
      encoder.setBuffer(
        buffers[workload == .independent ? pass : 0],
        offset: 0,
        index: 0)
      encoder.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
      )
      if let fence, pass + 1 != passCount {
        encoder.updateFence(fence)
      }
      encoder.endEncoding()
      if let event, pass + 1 != passCount {
        commandBuffer.encodeSignalEvent(event, value: nextEventValue)
        commandBuffer.encodeWaitForEvent(event, value: nextEventValue)
        nextEventValue += 1
      }
    }
    let encodeDuration = encodeStart.duration(to: .now)
    let commitStart = ContinuousClock.now
    commandBuffer.commit()
    let commitDuration = commitStart.duration(to: .now)
    commandBuffer.waitUntilCompleted()
    let wallDuration = wallStart.duration(to: .now)

    let expected: UInt32 = workload == .independent ? 1 : UInt32(passCount)
    let correct = buffers.allSatisfy {
      $0.contents().load(as: UInt32.self) == expected
    }
    let gpuDuration = max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
    return (
      Measurement(
        encodeMicroseconds: encodeDuration.microseconds,
        commitMicroseconds: commitDuration.microseconds,
        gpuMicroseconds: gpuDuration * 1_000_000,
        wallMicroseconds: wallDuration.microseconds
      ),
      correct
    )
  }
}

extension Duration {
  fileprivate var microseconds: Double {
    let c = components
    return Double(c.seconds) * 1_000_000
      + Double(c.attoseconds) / 1_000_000_000_000
  }
}

private final class CaseResult {
  let benchmarkCase: BenchmarkCase
  var results = [Measurement]()
  var allCorrect = true

  init(benchmarkCase: BenchmarkCase, capacity: Int) {
    self.benchmarkCase = benchmarkCase
    results.reserveCapacity(capacity)
  }
}

do {
  let benchmark = try Benchmark()
  let passCounts = [1, 10, 100, 500]
  let configurations: [(Workload, Tracking)] = [
    (.independent, .tracked),
    (.independent, .untracked),
    (.dependent, .tracked),
    (.dependent, .untrackedFenced),
    (.dependent, .untrackedEvent),
    // This intentionally unsafe case is useful as a correctness canary.
    (.dependent, .untracked),
  ]

  print("# Metal hazard tracking benchmark")
  print("# device=\(benchmark.device.name)")
  print("# os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
  print(
    "# columns: workload,tracking,passes,samples,correct,actual_resource_mode,encode_median_us,encode_p05_us,encode_p95_us,commit_median_us,commit_p05_us,commit_p95_us,gpu_median_us,gpu_p05_us,gpu_p95_us,wall_median_us"
  )

  for passCount in passCounts {
    let samples = passCount < 500 ? 200 : 80
    let caseResults = configurations.map { workload, tracking in
      CaseResult(
        benchmarkCase: benchmark.makeCase(
          workload: workload,
          tracking: tracking,
          passCount: passCount
        ),
        capacity: samples
      )
    }
    for result in caseResults {
      for _ in 0..<10 {
        _ = autoreleasepool { result.benchmarkCase.run() }
      }
    }
    for sample in 0..<samples {
      let indices =
        sample.isMultiple(of: 2)
        ? Array(caseResults.indices)
        : Array(caseResults.indices.reversed())
      for index in indices {
        let result = caseResults[index]
        let (measurement, correct) = autoreleasepool {
          result.benchmarkCase.run()
        }
        result.results.append(measurement)
        result.allCorrect = result.allCorrect && correct
      }
    }

    for result in caseResults {
      let benchmarkCase = result.benchmarkCase
      let encode = stats(result.results.map(\.encodeMicroseconds))
      let commit = stats(result.results.map(\.commitMicroseconds))
      let gpu = stats(result.results.map(\.gpuMicroseconds))
      let wall = stats(result.results.map(\.wallMicroseconds))
      print(
        [
          benchmarkCase.workload.rawValue,
          benchmarkCase.tracking.rawValue,
          String(passCount),
          String(samples),
          String(result.allCorrect),
          benchmarkCase.buffers[0].hazardTrackingMode == .tracked
            ? "tracked"
            : "untracked",
          String(format: "%.3f", encode.median),
          String(format: "%.3f", encode.p05),
          String(format: "%.3f", encode.p95),
          String(format: "%.3f", commit.median),
          String(format: "%.3f", commit.p05),
          String(format: "%.3f", commit.p95),
          String(format: "%.3f", gpu.median),
          String(format: "%.3f", gpu.p05),
          String(format: "%.3f", gpu.p95),
          String(format: "%.3f", wall.median),
        ].joined(separator: ","))
    }
  }
} catch {
  fputs("metal-hazard-bench: \(error)\n", stderr)
  exit(1)
}
