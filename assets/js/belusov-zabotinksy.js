function getColorIndicesForCoord(x, y, width) {
  // Retrieve r,b,g,alpha indices from continous image data arrray
  var red = y * (width * 4) + x * 4;
  return [red, red + 1, red + 2, red + 3];
}


function init(){

  // initialize canvas with random r,g,b value
  const width=600, height = 450;
  const alpha=1.2, beta=1., gamma=1.;
  var ctx = document.getElementById('sim').getContext('2d');
  var image = ctx.createImageData(width, height);
  const nestedArray = (width, height) => [new Float32Array(width*height), new Float32Array(width, height)];

  var p=0, q=1;
  var a = nestedArray(width, height);
  var b = nestedArray(width, height);
  var c = nestedArray(width, height);

  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      var indices = getColorIndicesForCoord(i, j, image.width);
      image.data[indices[3]] = 255;

      a[p][j*width+i] = Math.random();
      b[p][j*width+i] = Math.random();
      c[p][j*width+i] = Math.random();
    }
  }
  //var intervalID = setInterval(simulate, 10, ctx, image, kernel, width, height);
}

function simulate(ctx, image, width, height, alpha=1.2, beta=1, gamma=1){
  image = convolution2d(ctx, image, kernel, width, height);
  image = update(image, width, height);
  ctx.putImageData(image, 0, 0);
}
init();
