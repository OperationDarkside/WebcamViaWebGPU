//var streamRunning = false;
var cubeTexture;
var video;

function start() {
  video = document.querySelector("#videoElement");
  let btnStart = document.querySelector("#btnStart");

  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (stream) {
        //console.log(new VideoFrame(stream.getTracks()[0]).format);
        video.srcObject = stream;

        btnStart.disabled = false;
      })
      .catch(function (err0r) {
        console.log("Something went wrong!" + err0r);
      });
  }
}

function onBtnStartClick(btn) {
  let video = document.querySelector("#videoElement");
  video.play();

  btn.disabled = true;

  let btnStop = document.querySelector("#btnStop");
  btnStop.disabled = false;

  init();
}

function onBtnStopClick(btn) {
  let video = document.querySelector("#videoElement");
  video.pause();

  btn.disabled = true;

  let btnStart = document.querySelector("#btnStart");
  btnStart.disabled = false;
}


// Clear color for GPURenderPassDescriptor
const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };

// Vertex data for triangle
// Each vertex has 8 values representing position and color: X Y Z W R G B A

/*
const vertices = new Float32Array([
  0.0,  0.6, 0, 1, 1, 0, 0, 1,
 -0.5, -0.6, 0, 1, 0, 1, 0, 1,
  0.5, -0.6, 0, 1, 0, 0, 1, 1
]);
*/

const verticeeeees = new Float32Array([
  -1.0, 1.0, 0, 1, 0, 0,
  -1.0, -1.0, 0, 1, 0, 1.0,
  1.0, 1.0, 0, 1, 1.0, 0,
  1.0, 1.0, 0, 1, 1.0, 0,
  -1.0, -1.0, 0, 1, 0, 1.0,
  1.0, -1.0, 0, 1, 1.0, 1.0
]);

// Vertex and fragment shaders
const shaders2 = `
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var prevTexture: texture_external;
@group(0) @binding(2) var currTexture: texture_external;
@group(0) @binding(3) var nextTexture: texture_external;

// Uniforms to pass texture dimensions (needed for calculating neighbor UVs)
struct FrameDimensions {
  width: f32,
  height: f32,
};
@group(0) @binding(4) var<uniform> frameDims: FrameDimensions;


struct VertexOut {
  @builtin(position) position : vec4f,
  // @location(0) color : vec4f, // Color pass-through isn't needed
  @location(1) fragUV : vec2f,
}

@vertex
fn vertex_main(@location(0) position: vec4f,
               @location(1) uv : vec2f) -> VertexOut // Removed unused color input
{
  var output : VertexOut;
  output.position = position;
  output.fragUV = uv;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  // Calculate the size of one pixel in UV coordinates
  let texelSize = vec2f(1.0 / frameDims.width, 1.0 / frameDims.height);
  let centerUV = fragData.fragUV;

  var accumulatedColor = vec4f(0.0);
  var sampleCount = 0.0;

  // --- Spatial Kernel (3x3 Box Blur) ---
  for (var y: i32 = -1; y <= 1; y = y + 1) {
    for (var x: i32 = -1; x <= 1; x = x + 1) {
      let offset = vec2f(f32(x), f32(y)) * texelSize;
      let sampleUV = centerUV + offset;

      // --- Temporal Averaging ---
      // Sample the neighborhood pixel in all three frames
      // Using textureSampleBaseClampToEdge to handle boundaries automatically
      let prevColor = textureSampleBaseClampToEdge(prevTexture, mySampler, sampleUV);
      let currColor = textureSampleBaseClampToEdge(currTexture, mySampler, sampleUV);
      let nextColor = textureSampleBaseClampToEdge(nextTexture, mySampler, sampleUV);

      // Accumulate colors
      // Note: texture_external often contains YCbCr data. If so, proper
      // color space conversion should happen here or before averaging.
      // Assuming RGB for simplicity based on original shader.
      accumulatedColor += prevColor + currColor + nextColor;
      sampleCount += 3.0; // We added 3 samples
    }
  }

  // Calculate the average color
  let finalColor = accumulatedColor / sampleCount;

  // Return the averaged color, ensuring alpha is 1.0
  return vec4f(finalColor.rgb, 1.0);
}
`;

const shaders3 = `
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var prevTexture: texture_external;
@group(0) @binding(2) var currTexture: texture_external;
@group(0) @binding(3) var nextTexture: texture_external;

// Uniforms for dimensions and motion detection threshold
struct FrameInfo {
  size: vec2f,
  // How much difference constitutes motion? Adjust based on video content.
  // Value is based on luminance difference (0.0 to 1.0 range).
  // Start with something like 0.05 or 0.1
  motionThreshold: f32,
  // Optional: Add padding if needed for uniform buffer layout rules,
  // though often not strictly necessary for just 3 floats.
  @size(4) padding: f32 // Uncomment if layout issues occur
};
@group(0) @binding(4) var<uniform> frameInfo: FrameInfo;


struct VertexOut {
  @builtin(position) position : vec4f,
  // @location(0) color : vec4f, // Color pass-through isn't needed
  @location(1) fragUV : vec2f,
}

@vertex
fn vertex_main(@location(0) position: vec4f,
               @location(1) uv : vec2f) -> VertexOut // Removed unused color input
{
  var output : VertexOut;
  output.position = position;
  output.fragUV = uv;
  return output;
}

// Helper to calculate luminance (perceptual brightness)
fn luminance(color: vec3f) -> f32 {
  // Standard Rec. 709 luminance weights
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
  // Simpler approximation: dot(color, vec3f(0.299, 0.587, 0.114));
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  let centerUV = fragData.fragUV;
  let texelSize = vec2f(1.0 / frameInfo.size.x, 1.0 / frameInfo.size.y);

  // --- 1. Sample Center Pixels for Motion Detection ---
  // Note: Assuming textureSampleBaseClampToEdge gives RGB.
  // If YCbCr, you'd convert to RGB first or calculate difference in Y space.
  let prevCenterColor = textureSampleBaseClampToEdge(prevTexture, mySampler, centerUV).rgb;
  let currCenterColor = textureSampleBaseClampToEdge(currTexture, mySampler, centerUV).rgb;
  let nextCenterColor = textureSampleBaseClampToEdge(nextTexture, mySampler, centerUV).rgb;

  // --- 2. Calculate Motion Metric ---
  let lumPrev = luminance(prevCenterColor);
  let lumCurr = luminance(currCenterColor);
  let lumNext = luminance(nextCenterColor);

  // Total absolute difference from current frame's luminance
  let motionMetric = abs(lumCurr - lumPrev) + abs(lumCurr - lumNext);

  // --- 3. Perform Spatial Blurring (on all 3 frames for potential use) ---
  var blurredPrev = vec3f(0.0);
  var blurredCurr = vec3f(0.0);
  var blurredNext = vec3f(0.0);
  let spatialKernelRadius = 1; // For a 3x3 kernel (-1 to +1)
  var spatialWeightSum = 0.0;

  for (var y: i32 = -spatialKernelRadius; y <= spatialKernelRadius; y = y + 1) {
    for (var x: i32 = -spatialKernelRadius; x <= spatialKernelRadius; x = x + 1) {
      let offset = vec2f(f32(x), f32(y)) * texelSize;
      let sampleUV = centerUV + offset;
      let weight = 1.0; // Simple box blur weight

      blurredPrev += textureSampleBaseClampToEdge(prevTexture, mySampler, sampleUV).rgb * weight;
      blurredCurr += textureSampleBaseClampToEdge(currTexture, mySampler, sampleUV).rgb * weight;
      blurredNext += textureSampleBaseClampToEdge(nextTexture, mySampler, sampleUV).rgb * weight;
      spatialWeightSum += weight;
    }
  }
  // Normalize the spatial blur results
  blurredPrev /= spatialWeightSum;
  blurredCurr /= spatialWeightSum;
  blurredNext /= spatialWeightSum;


  // --- 4. Choose Smoothing based on Motion ---
  var finalColor = vec3f(0.0);

  if (motionMetric < frameInfo.motionThreshold) {
      // STATIC: Apply strong temporal smoothing (average the spatially blurred frames)
      finalColor = (blurredPrev + blurredCurr + blurredNext) / 3.0;
      // Alternative static: weighted average giving current frame more importance
      // finalColor = blurredPrev * 0.2 + blurredCurr * 0.6 + blurredNext * 0.2;
  } else {
      // MOTION: Use primarily the spatially blurred *current* frame
      finalColor = blurredCurr;
      // Alternative motion: slight blend with others
      // finalColor = blurredPrev * 0.05 + blurredCurr * 0.9 + blurredNext * 0.05;
  }

  // Using select (often faster on GPU than if/else, though compilers optimize)
  // let staticColor = (blurredPrev + blurredCurr + blurredNext) / 3.0;
  // let motionColor = blurredCurr;
  // finalColor = select(motionColor, staticColor, motionMetric < frameInfo.motionThreshold);


  return vec4f(finalColor, 1.0);
}
`;

const shaders4 = `
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var prevTexture: texture_external;
@group(0) @binding(2) var currTexture: texture_external;
@group(0) @binding(3) var nextTexture: texture_external;

struct VertexOut {
  @builtin(position) position : vec4f,
  // @location(0) color : vec4f, // Color pass-through isn't needed
  @location(1) fragUV : vec2f,
}

@vertex
fn vertex_main(@location(0) position: vec4f,
               @location(1) uv : vec2f) -> VertexOut // Removed unused color input
{
  var output : VertexOut;
  output.position = position;
  output.fragUV = uv;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  // Threshold for outlier rejection (tune this parameter)
  const OUTLIER_THRESHOLD = 0.15; // Adjust based on noise level
  let sampleUV = fragData.fragUV;

  // --- Temporal Averaging ---
  // Sample the neighborhood pixel in all three frames
  // Using textureSampleBaseClampToEdge to handle boundaries automatically
  let prevColor = textureSampleBaseClampToEdge(prevTexture, mySampler, sampleUV).rgb;
  let currColor = textureSampleBaseClampToEdge(currTexture, mySampler, sampleUV).rgb;
  let nextColor = textureSampleBaseClampToEdge(nextTexture, mySampler, sampleUV).rgb;

  let meanColor = (prevColor + currColor + nextColor) / 3.0;

  var samples = array<vec3f, 3>();
  samples[0] = prevColor;
  samples[1] = currColor;
  samples[2] = nextColor;

  // Count valid samples and sum them for averaging
  var validColorSum = vec3f(0.0);
  var validCount = 0.0;
  
  // Check each sample against the mean to detect outliers
  for (var i = 0; i < 3; i++) {
      // Calculate distance from mean (using luminance or color distance)
      let distance = length(samples[i] - meanColor);
      
      // If the sample is close enough to the mean, include it
      if (distance < OUTLIER_THRESHOLD) {
          validColorSum += samples[i];
          validCount += 1.0;
      }
  }
  
  // Protect against the case where all samples are rejected
  var finalColor = currColor;
  if (validCount > 0.0) {
      // Average of valid samples
      finalColor = validColorSum / validCount;
  }

  // Return the averaged color, ensuring alpha is 1.0
  return vec4f(finalColor, 1.0);
}
`;

// Main function

async function init() {
  // 1: request adapter and device
  if (!navigator.gpu) {
    throw Error('WebGPU not supported.');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw Error('Couldn\'t request WebGPU adapter.');
  }

  let device = await adapter.requestDevice();

  // 2: Create a shader module from the shaders template literal
  const shaderModule = device.createShaderModule({
    code: shaders4
  });

  // 3: Get reference to the canvas to render on
  const canvas = document.querySelector('#gpuCanvas');
  const context = canvas.getContext('webgpu');

  context.configure({
    device: device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: 'premultiplied'
  });

  // 4: Create vertex buffer to contain vertex data
  const vertexBuffer = device.createBuffer({
    size: verticeeeees.byteLength, // make it big enough to store vertices in
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // Copy the vertex data over to the GPUBuffer using the writeBuffer() utility function
  device.queue.writeBuffer(vertexBuffer, 0, verticeeeees, 0, verticeeeees.length);

  // --- Create the Uniform Buffer ---
  // Need space for 3 floats (width, height, threshold)
  // vec2f is 8 bytes, f32 is 4 bytes. Total = 12 bytes.
  // WGSL requires alignment, vec2f followed by f32 usually aligns correctly (8 + 4 = 12).
  // If adding more, be mindful of std140 layout rules (check WGSL spec).
  const frameInfoBufferSize = 4 * Float32Array.BYTES_PER_ELEMENT;
  const frameInfoBuffer = device.createBuffer({
    label: "Frame Info Uniform Buffer",
    size: frameInfoBufferSize, // e.g., 12 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 5: Create a GPUVertexBufferLayout and GPURenderPipelineDescriptor to provide a definition of our render pipline
  const vertexBuffers = [{
    attributes: [{
      shaderLocation: 0, // position
      offset: 0,
      format: 'float32x4'
    }, {
      shaderLocation: 1, // uv
      offset: 16,
      format: 'float32x2'
    }],
    arrayStride: 24,
    stepMode: 'vertex'
  }];

  const pipelineDescriptor = {
    vertex: {
      module: shaderModule,
      entryPoint: 'vertex_main',
      buffers: vertexBuffers
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragment_main',
      targets: [{
        format: navigator.gpu.getPreferredCanvasFormat()
      }]
    },
    primitive: {
      topology: 'triangle-list'
    },
    layout: 'auto'
  };

  // Create a sampler with linear filtering for smooth interpolation.
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // 6: Create the actual render pipeline

  const renderPipeline = device.createRenderPipeline(pipelineDescriptor);

  //var prevprevVideoFrame = new VideoFrame(video);
  var prevVideoFrame = new VideoFrame(video);
  var currVideoFrame = new VideoFrame(video);
  var nextVideoFrame = new VideoFrame(video);

  let videoWidth = 0;
  let videoHeight = 0;

  // --- Get initial dimensions (important!) ---
  // Wait for video metadata or first frame to get dimensions
  video.addEventListener('loadedmetadata', () => {
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
    console.log("Video dimensions:", videoWidth, videoHeight);
    updateDimensionsUniform(); // Call a function to update the buffer
  }, { once: true }); // Run only once

  // --- Update the Bind Group Layout ---
  // Ensure binding 4 expects a buffer:
  // { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }

  // --- Update the Bind Group ---
  // Ensure entry 4 points to frameInfoBuffer:
  // { binding: 4, resource: { buffer: frameInfoBuffer } }

  // --- Write Data to the Buffer (in your loop or when needed) ---
  const motionThresholdValue = 0.08; // *** TUNABLE PARAMETER ***
  // Function to update the uniform buffer
  function updateDimensionsUniform() {
    if (videoWidth > 0 && videoHeight > 0) {
      // Data order must match the WGSL struct: size (vec2f), motionThreshold (f32)
      const frameInfoData = new Float32Array([videoWidth, videoHeight, motionThresholdValue]);
      device.queue.writeBuffer(
        frameInfoBuffer, 0, frameInfoData.buffer,
        frameInfoData.byteOffset, frameInfoData.byteLength
      );
    } else {
      // Fallback if dimensions still aren't ready (shouldn't happen after loadedmetadata)
      const frameInfoData = new Float32Array([1.0, 1.0, motionThresholdValue]);
      device.queue.writeBuffer(
        frameInfoBuffer, 0, frameInfoData.buffer,
        frameInfoData.byteOffset, frameInfoData.byteLength
      );
    }
  }

  // Initial update attempt (might be 0x0 before 'loadedmetadata')
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  updateDimensionsUniform();


  function frame() {
    // Sample is no longer the active page.
    //if (!pageState.active) return;
    try {
      // 1. Update VideoFrame references
      // The *previous* nextVideoFrame becomes the new currVideoFrame
      // The *previous* currVideoFrame becomes the new prevVideoFrame
      // We discard the old prevVideoFrame implicitly (JS garbage collection)
      const prevprevVideoFrame = prevVideoFrame;
      prevprevVideoFrame.close(); // Close the old frame to free resources

      prevVideoFrame = currVideoFrame;
      currVideoFrame = nextVideoFrame;

      // 2. Get the *new* next frame
      // IMPORTANT: Ensure video is playing and ready. Might need error handling/checks.
      if (video.readyState >= video.HAVE_CURRENT_DATA) { // Check if frame data is available
        nextVideoFrame = new VideoFrame(video); // Create the *next* frame snapshot
      } else {
        console.warn("Video not ready for new frame.");
        // Decide how to handle this: skip frame, reuse last frame?
        // Reusing the last 'next' might be okay visually for a short time.
        // If nextVideoFrame is null here, skip rendering or handle below.
        requestAnimationFrame(frame); // Try again next anim frame
        return; // Exit this frame function
      }


      // 3. Check if we have all necessary frames
      if (!prevVideoFrame || !currVideoFrame || !nextVideoFrame) {
        console.log("Waiting for frames to buffer...");
        // Request the next frame and wait
        if ('requestVideoFrameCallback' in video) {
          video.requestVideoFrameCallback(frame);
        } else {
          requestAnimationFrame(frame);
        }
        return; // Not ready to render yet
      }

      // Check and update dimensions if they changed (less common but possible)
      if (video.videoWidth !== videoWidth || video.videoHeight !== videoHeight) {
        videoWidth = video.videoWidth;
        videoHeight = video.videoHeight;
        updateDimensionsUniform();
        console.log("Video dimensions changed:", videoWidth, videoHeight);
      }

      const uniformBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
          {
            binding: 1,
            resource: device.importExternalTexture({
              source: prevVideoFrame,
            }),
          },
          {
            binding: 2,
            resource: device.importExternalTexture({
              source: currVideoFrame,
            }),
          },
          {
            binding: 3,
            resource: device.importExternalTexture({
              source: nextVideoFrame,
            }),
          },
          /* { // 4: Frame Dimensions Uniform Buffer <--- NEW ENTRY
            binding: 4,
            resource: {
              buffer: frameInfoBuffer
              // Optionally specify offset/size if using part of a larger buffer
              // offset: 0, (default)
              // size: frameDimensionsBufferSize (default is buffer size)
            }
          } */
        ],
      });

      // 7: Create GPUCommandEncoder to issue commands to the GPU
      // Note: render pass descriptor, command encoder, etc. are destroyed after use, fresh one needed for each frame.
      const commandEncoder = device.createCommandEncoder();
      const textureView = context.getCurrentTexture().createView();

      // 8: Create GPURenderPassDescriptor to tell WebGPU which texture to draw into, then initiate render pass
      const renderPassDescriptor = {
        colorAttachments: [
          {
            view: textureView,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      };

      const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
      // 9: Draw
      passEncoder.setPipeline(renderPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, vertexBuffer);
      passEncoder.draw(6, 1, 0, 0);
      passEncoder.end();
      // 10: End frame by passing array of command buffers to command queue for execution
      device.queue.submit([commandEncoder.finish()]);

      if ('requestVideoFrameCallback' in video) {
        video.requestVideoFrameCallback(frame);
      } else {
        requestAnimationFrame(frame);
      }
    } catch (error) {
      console.error(error);
    }
  }

  if ('requestVideoFrameCallback' in video) {
    video.requestVideoFrameCallback(frame);
  } else {
    requestAnimationFrame(frame);
  }
}