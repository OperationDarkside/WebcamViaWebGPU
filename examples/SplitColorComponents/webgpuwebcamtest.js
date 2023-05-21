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

async function onBtnStartClick(btn) {
    let video = document.querySelector("#videoElement");
    await video.play();

    const canvas = document.querySelector('#gpuCanvas');
    canvas.width = video.videoWidth * 2;
    canvas.height = video.videoHeight * 2;

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
    // Red
    -1.0,  1.0, 0, 0, 0, 0,
   -1.0, 0, 0, 0, 1.0, 0,
    0, 1.0, 0, 1.0, 0, 0,
    0, 1.0, 0, 1.0, 0, 0,
    -1.0, 0, 0, 0, 1.0, 0,
    0, 0, 0, 1.0, 1.0, 0,
    // Green
    0, 1.0, 0, 0, 0, 1,
    0, 0, 0, 0, 1.0, 1,
    1.0, 1.0, 0, 1.0, 0, 1,
    1.0, 1.0, 0, 1.0, 0, 1,
    0, 0, 0, 0, 1.0, 1,
    1.0, 0, 0, 1.0, 1.0, 1,
    // Blue
    -1.0, 0, 0, 0, 0, 2,
    -1.0, -1.0, 0, 0, 1.0, 2,
    0, 0, 0, 1.0, 0, 2,
    0, 0, 0, 1.0, 0, 2,
    -1.0, -1.0, 0, 0, 1.0, 2,
    0, -1.0, 0, 1.0, 1.0, 2,
    // Lumi
    0, 0, 0, 0, 0, 3,
    0, -1.0, 0, 0, 1.0, 3,
    1.0, 0, 0, 1.0, 0, 3,
    1.0, 0, 0, 1.0, 0, 3,
    0, -1.0, 0, 0, 1.0, 3,
    1.0, -1.0, 0, 1.0, 1.0, 3
]);

// Vertex and fragment shaders

const shaders = `
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var prevTexture: texture_external;
@group(0) @binding(2) var currTexture: texture_external;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) index : f32,
  @location(2) fragPosition: vec4<f32>,
}

@vertex
fn vertex_main(@location(0) position: vec3f,
               @location(1) uv : vec2f,
               @location(2) ind : f32) -> VertexOut
{
  var output : VertexOut;
  output.position = vec4f(position, 1.0);
  output.fragUV = uv;
  output.index = ind;
  output.fragPosition = 0.5 * (output.position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
    let prevColor = textureSampleBaseClampToEdge(prevTexture, mySampler, fragData.fragUV); // * fragData.fragPosition;
    let currColor = textureSampleBaseClampToEdge(currTexture, mySampler, fragData.fragUV); // * fragData.fragPosition;
    var outColor: vec4f;
    //if(any(prevColor != currColor)){
    //if((prevColor[0] == currColor[0]) && (prevColor[1] == currColor[1]) && (prevColor[2] == currColor[2])){
      // outColor = vec4f(1.0, 1.0, 1.0, 1.0);
      //outColor = currColor;
    //}
    // return outColor;

    let ind = i32(fragData.index);
    switch ind {
      case 0, {
        outColor = vec4f(currColor[0], 0, 0, 1.0);
      }
      case 1, {
        outColor = vec4f(0, currColor[1], 0, 1.0);
      }
      case 2, {
        outColor = vec4f(0, 0, currColor[2], 1.0);
      }
      case 3, {
        var gs = dot(currColor.xyz, vec3f(0.21, 0.71, 0.07));
        outColor = vec4f(vec3f(gs), 1.0);
      }
      default {
        outColor = vec4f(1.0, 1.0, 0.0, 1.0);
      }
    }

    return outColor;

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
    code: shaders
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

  // 5: Create a GPUVertexBufferLayout and GPURenderPipelineDescriptor to provide a definition of our render pipline
  const vertexBuffers = [{
    attributes: [{
      shaderLocation: 0, // position
      offset: 0,
      format: 'float32x3'
    }, {
        shaderLocation: 1, // uv
        offset: 12,
        format: 'float32x2'
    }, {
      shaderLocation: 2, // index
      offset: 20,
      format: 'float32'
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

  var prevprevVideoFrame = new VideoFrame(video);
  var prevVideoFrame = new VideoFrame(video);
  var currVideoFrame = new VideoFrame(video);

  function frame() {
    // Sample is no longer the active page.
    //if (!pageState.active) return;

    // Manage video frames
    prevprevVideoFrame.close();
    prevprevVideoFrame = prevVideoFrame;
    prevVideoFrame = currVideoFrame;
    currVideoFrame = new VideoFrame(video);

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
        }
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
    passEncoder.draw(24, 1, 0, 0);
    passEncoder.end();
    // 10: End frame by passing array of command buffers to command queue for execution
    device.queue.submit([commandEncoder.finish()]);

    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback(frame);
    } else {
      requestAnimationFrame(frame);
    }
  }

    if ('requestVideoFrameCallback' in video) {
        video.requestVideoFrameCallback(frame);
    } else {
        requestAnimationFrame(frame);
    }
}