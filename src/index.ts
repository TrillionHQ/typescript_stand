import * as tf from '@tensorflow/tfjs';

// Utility to normalize tensor values between 0 and 1
function minMaxNormalize(data: tf.Tensor): tf.Tensor {
    const min = tf.min(data);
    const max = tf.max(data);
    return data.sub(min).div(max.sub(min));
}

// Preprocess the image
function preprocessImage(imageElement: HTMLImageElement): tf.Tensor {
    if (!imageElement) {
        throw new Error("Image element is not available");
    }
    let tensor = tf.browser.fromPixels(imageElement)
        .resizeBilinear([512, 512])
        .toFloat()
        .div(255.0)
        .expandDims();
    return tensor;
}

// Create a rectangular convolution kernel
function createRectangularKernel(height: number, width: number, angleDeg: number): tf.Tensor {
    const angle = angleDeg * (Math.PI / 180);
    const center = [Math.floor(height / 2), Math.floor(width / 2)];
    const y = tf.range(0, height).expandDims(1).tile([1, width]);
    const x = tf.range(0, width).expandDims(0).tile([height, 1]);
    const xRot = x.sub(center[1]).mul(tf.cos(angle)).sub(y.sub(center[0]).mul(tf.sin(angle)));
    const yRot = x.sub(center[1]).mul(tf.sin(angle)).add(y.sub(center[0]).mul(tf.cos(angle)));
    const kernel = xRot.abs().less(width / 2).logicalAnd(yRot.abs().less(height / 2)).toFloat();
    return kernel.div(tf.sum(kernel));
}

// Improved edge detection based on convolution
function findFingerEdges(conv: tf.Tensor2D, pointX: number, pointY: number, cosAngle: number, sinAngle: number, threshold: number, maxDistance: number): [number, number] | null {
    for (let i = 0; i < maxDistance; i++) {
        const x = Math.round(pointX + i * cosAngle);
        const y = Math.round(pointY + i * sinAngle);

        if (x < 0 || x >= conv.shape[1] || y < 0 || y >= conv.shape[0]) {
            break;
        }

        const valueTensor = conv.slice([y, x], [1, 1]);
        const value = valueTensor.dataSync()[0];
        valueTensor.dispose();

        if (value > threshold) {
            return [x, y];
        }
    }
    return null;
}

// Display original image on canvas
function displayOriginalImage(imageElement: HTMLImageElement, canvasId: string): void {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
    if (!canvas) {
        console.error("Canvas element not found");
        return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error("Could not get 2D context from canvas");
        return;
    }
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
}


// Compute the width of the finger using convolution with the new kernel
function findFingerWidth(prediction: tf.Tensor4D, pointX: number, pointY: number, angleDeg: number, kernelHeight: number = 15, kernelWidth: number = 7, threshold: number = 0.1, maxDistance: number = 50): [[number, number] | null, [number, number] | null] {
    const perpAngle = angleDeg + 90;
    const cosPerp = Math.cos(perpAngle * (Math.PI / 180));
    const sinPerp = Math.sin(perpAngle * (Math.PI / 180));

    const leftKernel = createRectangularKernel(kernelHeight, kernelWidth, perpAngle) as tf.Tensor2D;
    const rightKernel = tf.reverse2d(leftKernel);

    const convLeft = tf.conv2d(prediction as tf.Tensor4D, leftKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D, 1, 'same').squeeze();
    const convRight = tf.conv2d(prediction as tf.Tensor4D, rightKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D, 1, 'same').squeeze();

    const leftEdge = findFingerEdges(convLeft as tf.Tensor2D, pointX, pointY, -cosPerp, -sinPerp, threshold, maxDistance);
    const rightEdge = findFingerEdges(convRight as tf.Tensor2D, pointX, pointY, cosPerp, sinPerp, threshold, maxDistance);

    convLeft.dispose();
    convRight.dispose();
    leftKernel.dispose();
    rightKernel.dispose();

    return [leftEdge, rightEdge];
}

// Display results and overlay on the canvas
function postprocessAndDisplay(tensor: tf.Tensor, imageCanvasId: string, overlayCanvasId: string, pointX: number, pointY: number, angleDeg: number, leftEdge: [number, number] | null, rightEdge: [number, number] | null): void {
    const imageCanvas = document.getElementById(imageCanvasId) as HTMLCanvasElement | null;
    if (!imageCanvas) {
        console.error("Image canvas not found");
        return;
    }
    const overlayCanvas = document.getElementById(overlayCanvasId) as HTMLCanvasElement | null;
    if (!overlayCanvas) {
        console.error("Overlay canvas not found");
        return;
    }
    const overlayCtx = overlayCanvas.getContext('2d');
    if (!overlayCtx) {
        console.error("Could not get 2D context from overlay canvas");
        return;
    }

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const grayTensor = tensor.squeeze().clipByValue(0, 1);
    tf.browser.toPixels(grayTensor as tf.Tensor3D, imageCanvas).then(() => {
        overlayCtx.beginPath();
        if (leftEdge && rightEdge) {
            overlayCtx.moveTo(leftEdge[0], leftEdge[1]);
            overlayCtx.lineTo(rightEdge[0], rightEdge[1]);
            overlayCtx.strokeStyle = 'red';
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();
        }

        overlayCtx.beginPath();
        overlayCtx.arc(pointX, pointY, 5, 0, 2 * Math.PI);
        overlayCtx.fillStyle = 'green';
        overlayCtx.fill();
    });
}

// Calculate and log the finger width
function calculateFingerWidth(leftEdge: [number, number] | null, rightEdge: [number, number] | null): void {
    if (leftEdge && rightEdge) {
        const width = Math.sqrt(Math.pow(leftEdge[0] - rightEdge[0], 2) + Math.pow(leftEdge[1] - rightEdge[1], 2));
        console.log(`Finger width: ${width.toFixed(2)} pixels`);
    } else {
        console.log('Could not detect finger width');
    }
}

// Main function to execute the logic
async function run() {
    const model = await tf.loadGraphModel('Daniils_sept_6/model.json');
    console.log('Model loaded successfully.');
    const imageElement = document.createElement('img');
    imageElement.src = 'hand.jpg'; // Path to your image file
    document.body.appendChild(imageElement);

    imageElement.onload = async () => {
        displayOriginalImage(imageElement, 'originalCanvas');
        const tensor = preprocessImage(imageElement);
        const prediction = model.predict(tensor);
        const output = (prediction as tf.NamedTensorMap)[2] as tf.Tensor;

        const normalizedPrediction = minMaxNormalize(output) as tf.Tensor4D;

        const points = [[282, 179], [272, 234], [244, 276]];
        const angleDeg = 35;

        points.forEach(([pointX, pointY]) => {
            const [leftEdge, rightEdge] = findFingerWidth(normalizedPrediction, pointX, pointY, angleDeg);
            postprocessAndDisplay(normalizedPrediction, 'imageCanvas', 'overlayCanvas', pointX, pointY, angleDeg, leftEdge, rightEdge);
            calculateFingerWidth(leftEdge, rightEdge);
        });
    };
}

run().catch(console.error);
