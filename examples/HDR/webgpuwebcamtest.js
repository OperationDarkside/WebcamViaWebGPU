//var streamRunning = false;
var cubeTexture;
let video;
let imageCapture;

const constraints = {
  width: { min: 640, ideal: 1280 },
  height: { min: 480, ideal: 720 },
  iso : 50
};

function start() {
    video = document.querySelector("#videoElement");
    let btnStart = document.querySelector("#btnStart");

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(async function (stream) {
            //console.log(new VideoFrame(stream.getTracks()[0]).format);
            video.srcObject = stream;

            const track = stream.getVideoTracks()[0];
            await track.applyConstraints(constraints);
            imageCapture = new ImageCapture(track);


            const photoCapa = await imageCapture.getPhotoCapabilities();
            console.log(JSON.stringify(photoCapa));
            //console.log(track.getSupportedConstraints());
            //console.log("--------------")
            const constr = await track.getConstraints();
            console.log(JSON.stringify(constr));
            const capa = track.getCapabilities();
            console.log(JSON.stringify(capa));
            
            btnStart.disabled = false;

            return imageCapture.getPhotoCapabilities();
        }).then((photoCapabilities) => {
          const capa = imageCapture.track.getCapabilities();
          console.log(capa);
        }).catch(function (err0r) {
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
    -1.0,  1.0, 0, 1, 1, 0, 0, 1, 0, 0,
   -1.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1.0,
    1.0, 1.0, 0, 1, 0, 0, 1, 1, 1.0, 0,
    1.0, 1.0, 0, 1, 0, 0, 1, 1, 1.0, 0,
    -1.0, -1.0, 0, 1, 0, 1, 0, 1, 0, 1.0,
    1.0, -1.0, 0, 1, 1, 0.5, 0.75, 1.0, 1.0, 1.0
]);

// Vertex and fragment shaders

const shaders = `
@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var prevTexture: texture_external;
@group(0) @binding(2) var currTexture: texture_external;

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) fragUV : vec2f,
}

@vertex
fn vertex_main(@location(0) position: vec4f,
               @location(1) color: vec4f,
               @location(2) uv : vec2f) -> VertexOut
{
  var output : VertexOut;
  output.position = position;
  output.color = color;
  output.fragUV = uv;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
    let prevColor = textureSampleBaseClampToEdge(prevTexture, mySampler, fragData.fragUV);
    let currColor = textureSampleBaseClampToEdge(currTexture, mySampler, fragData.fragUV);
    var outColor = vec4f(0.0, 0.0, 0.0, 1.0);
    if(any(prevColor != currColor)){
    //if((prevColor[0] == currColor[0]) && (prevColor[1] == currColor[1]) && (prevColor[2] == currColor[2])){
      // outColor = vec4f(1.0, 1.0, 1.0, 1.0);
      outColor = currColor;
    }
    return outColor;
    //return textureSampleBaseClampToEdge(myTexture, mySampler, fragData.fragUV);
    //return fragData.color;
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
      format: 'float32x4'
    }, {
      shaderLocation: 1, // color
      offset: 16,
      format: 'float32x4'
    }, {
        shaderLocation: 2, // uv
        offset: 32,
        format: 'float32x2'
    }],
    arrayStride: 40,
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
    passEncoder.draw(6, 1, 0, 0);
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