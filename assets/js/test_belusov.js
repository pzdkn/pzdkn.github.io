
function test_padding(image, width, height){
  // test padding
  const kernel = new Uint8ClampedArray(3*3);
  const kernelSize = Math.sqrt(kernel.length);
  const padSize = (kernelSize - 1) / 2;
  var paddedCtx = document.getElementById('padded').getContext('2d');
  var paddedImage = paddedCtx.createImageData(width+2*padSize, height+2*padSize);
  paddedImage = paddData(image, paddedImage, padSize, width, height);
  paddedCtx.putImageData(paddedImage, 0, 0);
  console.log("Padded Data")
  console.log(paddedImage.data)
}

function test_convolution(image, width, height){
  // test convolution
  const kernel = new Float32Array(3*3);
  const kernelSize = Math.sqrt(kernel.length);
  const padSize = (kernelSize - 1) / 2;
  console.log(`kernelSize : ${kernelSize}`);
  console.log(`padSize, ${padSize}`);
  var convCtx = document.getElementById('conv').getContext('2d');
  for (var k=0; k<kernel.length; k+=1){
    console.log("doing it");
    kernel[k] = 1./kernel.length;
  }
  console.log(`kernel : ${kernel}`);
  image = convolution2d(convCtx, image, kernel, width, height);
  convCtx.putImageData(image, 0, 0);
  console.log("Convoluted Data")
  console.log(image.data)
}
