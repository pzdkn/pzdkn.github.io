function getColorIndicesForCoord(x, y, width) {
  // Retrieve r,b,g,alpha indices from continous image data arrray
  var red = y * (width * 4) + x * 4;
  return [red, red + 1, red + 2, red + 3];
}

function paddData(image, paddedImage, padSize, width, height, padding="zero"){
  // Zero-Padding
  // Upper Horizontal
  const paddedData = paddedImage.data;
  const imageData = image.data;
  for (var i=0; i<width+2*padSize; i+=1){
    for (var j=0; j<padSize; j+=1){
      var indices = getColorIndicesForCoord(i, j, paddedImage.width);
      for (const index in indices.slice(0,3)){
        paddedData[index] = 0;
      }
      paddedData[indices[3]]=255; // set alpha to full
    }
  }
  // Lower Horizontal
  for (var i=0; i<width+2*padSize; i+=1){
    for (var j=height+padSize; j<height+2*padSize; j+=1){
      var indices = getColorIndicesForCoord(i, j, paddedImage.width);
      for (const index in indices.slice(0,3)){
        paddedData[index] = 0;
      }
      paddedData[indices[3]]=255; // set alpha to full
    }
  }
  // Left Vertical
  for (var i=0; i<padSize; i+=1){
    for (var j=padSize; j<height+padSize; j+=1){
      var indices = getColorIndicesForCoord(i, j, paddedImage.width);
      for (const index in indices.slice(0,3)){
        paddedData[index] = 0;
      }
      paddedData[indices[3]]=255; // set alpha to full
    }
  }
  // Right Vertical
  for (var i=width+padSize; i<width+2*padSize; i+=1){
    for (var j=padSize; j<height+padSize; j+=1){
      var indices = getColorIndicesForCoord(i, j, paddedImage.width);
      for (const index in indices.slice(0,3)){
        paddedData[index] = 0;
      }
      paddedData[indices[3]]=255; // set alpha to full
    }
  }
  // Copy original data into padded data
  for (var i=padSize; i<width+padSize; i+=1){
    for (var j=padSize; j<height+padSize; j+=1){
      var indicesPadded = getColorIndicesForCoord(i, j, paddedImage.width);
      var indicesVanilla = getColorIndicesForCoord(i-padSize, j-padSize, image.width);
      for (var k=0; k<4; k+=1){
        paddedData[indicesPadded[k]] = imageData[indicesVanilla[k]];
      }
    }
  }
  return paddedImage;
}

function convolution2d(ctx, image, kernel, width, height){
  // 2d convolution with kernel on channel in [r,g,b]
  var kernelSize = Math.sqrt(kernel.length);

  if (kernelSize % 1 != 0){
    throw 'kernel must be square!';
  }
  if (kernelSize % 2 == 0){
    throw 'kernel size must be odd';
  }
  var padSize = (kernelSize - 1) / 2;
  var paddedImage = ctx.createImageData(width+2*padSize, height+2*padSize);
  paddedImage = paddData(image, paddedImage, padSize, width, height);

  var first = false;
  // Iterate over padded data
  for (var i=padSize; i<width+padSize; i=i+1){ // x-axis
    for (var j=padSize; j<height+padSize; j+=1){ // y-axis
      let result = [0, 0, 0];
      // Do convolution
      for (var k=0; k<kernelSize; k+=1){ // x-axis
        for (var l=0; l<kernelSize; l+=1){ // y-axis
          const indicesPadd = getColorIndicesForCoord(i+k-padSize, j+l-padSize, paddedImage.width);
          // Do it for all channels
          for (var m=0; m<3; m+=1){
            result[m]+=kernel[l*kernelSize+k] * paddedImage.data[indicesPadd[m]];
          }
        }
      }

      // Store convoluted results into data array for each channel
      const indices = getColorIndicesForCoord(i-padSize, j-padSize, image.width);
      for (var m=0; m<3; m+=1){
        image.data[indices[m]] = result[m];
      }
    }
  }
  return image;
}

function update(image, width, height, alpha=1, beta=1, gamma=1){
  console.log("update");
  newData = new Uint8ClampedArray(4*width*height)
  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      const indices = getColorIndicesForCoord(i, j , image.width);
      newData[indices[0]] = image.data[indices[0]]
          +  image.data[indices[0]]
          * (alpha * image.data[indices[1]] - gamma * image.data[indices[2]]);
      newData[indices[1]] = image.data[indices[1]]
          + image.data[indices[1]]
          * (beta * image.data[indices[2]] - alpha * image.data[indices[0]]);
      newData[indices[2]] = image.data[indices[2]]
          + image.data[indices[2]]
          * (gamma * image.data[indices[0]] - beta * image.data[indices[1]]);
      newData[indices[3]] = image.data[indices[3]];
    }
  }
  image.data.set(newData);
  return image;
}

function init(){
  // initialize canvas with random r,g,b value
  const width=600, height = 450;
  var ctx = document.getElementById('sim').getContext('2d');
  var image = ctx.createImageData(width, height);
  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      var indices = getColorIndicesForCoord(i, j, image.width);
      image.data[indices[0]] = 255*Math.random();
      image.data[indices[1]] = 255*Math.random();
      image.data[indices[2]] = 255*Math.random();
      image.data[indices[3]] = 255;
    }
  }
  // init kernel
  const kernel = new Float32Array(3*3);
  for (var k=0; k<kernel.length; k+=1){
    kernel[k] = 1./kernel.length;
  }
  var intervalID = setInterval(simulate, 10, ctx, image, kernel, width, height);
}

function simulate(ctx, image, kernel, width, height, alpha=1.2, beta=1, gamma=1){
  image = convolution2d(ctx, image, kernel, width, height);
  image = update(image, width, height);
  ctx.putImageData(image, 0, 0);
}
init();
