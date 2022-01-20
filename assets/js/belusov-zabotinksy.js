function getColorIndicesForCoord(x, y, width) {
  // Retrieve r,b,g,alpha indices from continous image data arrray
  var red = y * (width * 4) + x * 4;
  return [red, red + 1, red + 2, red + 3];
}


function init(simulationForm){
  var {alpha:alpha, beta:beta, gamma:gamma, channels:channels} = simulationForm;
  console.log(`channels: ${channels}`);
  console.log(`alpha: ${alpha}, beta: ${beta}, gamma: ${gamma}`);
  // initialize canvas with random r,g,b value
  const width=600, height = 450;
  var ctx = document.getElementById('sim').getContext('2d');
  var image = ctx.createImageData(width, height);
  const nestedArray = (width, height) => [new Float32Array(width*height), new Float32Array(width*height)];

  var a = nestedArray(width, height);
  var b = nestedArray(width, height);
  var c = nestedArray(width, height);
  var array = [a, b, c];

  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      var indices = getColorIndicesForCoord(i, j, image.width);

      a[p][j*width+i] = Math.random();
      b[p][j*width+i] = Math.random();
      c[p][j*width+i] = Math.random();
      for (var k=0; k<channels; k+=1){
        image.data[indices[k]] = 255 * array[k][p][j*width+i];
      }
      image.data[indices[3]] = 255;
    }
  }
  var params = {"a":a, "b":b, "c": c,
                "width":width, "height":height,
                "alpha": alpha, "beta":beta, "gamma":gamma,
                "channels":channels};
  ctx.putImageData(image, 0, 0);
  var intervalID = setInterval(simulate, 100, ctx, image, params);
  return intervalID;
}

function simulate(ctx, image, params){
  var {a: a, b: b, c: c,
       width: width, height: height,
       alpha: alpha, beta: beta, gamma: gamma,
       channels: channels} = params;
  console.log(`channels: ${channels}`);
  var array = [a, b, c];
  const clamp = (num, min, max) => Math.max(Math.min(num, max), min);
  const mod = (n, m) => ((n % m) + m) % m;

  // not sure if changes to a, b, c persist across calls of simulate
  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      var c_a=0., c_b=0., c_c=0.;
      for (var k=i-1; k<=i+1; k+=1){
        for (var l=j-1; l<=j+1; l+=1){
          c_a += a[p][mod(l,height)*width+mod(k,width)];
          c_b += b[p][mod(l,height)*width+mod(k,width)];
          c_c += c[p][mod(l,height)*width+mod(k,width)];
        }
      }
      c_a /= 9.
      c_b /= 9.
      c_c /= 9.
      a[q][j * width + i] = clamp(c_a + alpha*(c_b- c_c), 0., 1.);
      b[q][j * width + i] = clamp(c_b + beta*(c_c- c_a), 0., 1.);
      c[q][j * width + i] = clamp(c_c + alpha*(c_a- c_b), 0., 1.);
    }
  }
  for (var k=0; k<channels; k+=1){
    image = setImageData(image, array[k][q], k, width, height);
  }

  ctx.putImageData(image, 0, 0);
  if (p == 0) {
    p = 1, q = 0;
  } else {
    p = 0, q = 1;
  }
}

function setImageData(image, channelData, channelIndex, width, height){
  for (var i=0; i<width; i+=1){
    for (var j=0; j<height; j+=1){
      const imageIndices = getColorIndicesForCoord(i, j, width);
      image.data[imageIndices[channelIndex]] = 255*channelData[j*width+i];
    }
  }
  return image;
}

var p=0, q=1;
var first = true;
var simulationForm = {alpha:1.3, beta:1., gamma:1., channels:3};
document.addEventListener("DOMContentLoaded", function(){
  var intervalID = init(simulationForm);
  var loginForm = document.getElementById("simulation-form");
  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    var formData = new FormData(e.target);
    var simulationForm = Object.fromEntries(formData);
    clearInterval(intervalID);
    intervalID = init(simulationForm);
  });
}, false)
