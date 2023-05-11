# WebGPUTesting

A short demonstration of how to display a real-time webcam feed in a canvas via a WebGPU renderpipeline.

The roundabout way via the video element is because I didn't know better. I'm a JS newb after all.

You can use this code to test image/video manipulation shaders.

## How to use it:

1. Open the [main.html](main.html) in a WebGPU-capable browser (Chrome 113 at the time of writing)
2. Allow access to a video source of yours (might take a few seconds)
3. Press the "Start" button
4. You should see your video feed in the video element on top and the WebGPU rendered version below the buttons

## Examples

1. Only Show motion
  Computes a diff between the previous and current video frame and only displays pixels from the current frame, if they differ from the pixel at the same position in the previous frame.
  This one requires a chrome developer feature flag to be enabled. Look at the console for its name, when the code throws.
  !!! With a noisy video feed, e.g. cheap sensor and low light conditions, the diff can look almost identical to the raw feed.

### Sidenotes for beginners:

- The WebGPU rendering can only start, when the video element has a valid source. Otherwise the following code will throw:
```javascript
resource: device.importExternalTexture({
  source: video,
})
```

- The fragment shader uses the "texture_external" type and the "textureSampleBaseClampToEdge" sampling function. I can't exactly tell why those are needed. My guess is, that external textures have a special format, that need to be handled differently. I copied the code from the [Uploading Video WebGPU](https://webgpu.github.io/webgpu-samples/samples/videoUploading#../../shaders/sampleExternalTexture.wgsl) example.
