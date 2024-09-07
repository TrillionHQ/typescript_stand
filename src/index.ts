import * as tf from '@tensorflow/tfjs';

function createRectangularKernel(height: number, width: number, angleDeg: number): tf.Tensor {
  const angleRad = angleDeg * (Math.PI / 180); // перевод градусов в радианы

  // Создание пустого ядра и определение центра
  const centerY = Math.floor(height / 2);
  const centerX = Math.floor(width / 2);

  const cosAngle = Math.cos(angleRad);
  const sinAngle = Math.sin(angleRad);

  // Создание координатной сетки
  const yCoords = tf.range(0, height, 1, 'int32');
  const xCoords = tf.range(0, width, 1, 'int32');
  const yGrid = tf.tile(yCoords.reshape([height, 1]), [1, width]);
  const xGrid = tf.tile(xCoords.reshape([1, width]), [height, 1]);

  // Применение поворота
  const xRot = tf.add(
    tf.mul(tf.sub(xGrid, centerX), cosAngle),
    tf.mul(tf.sub(yGrid, centerY), -sinAngle)
  );
  const yRot = tf.add(
    tf.mul(tf.sub(xGrid, centerX), sinAngle),
    tf.mul(tf.sub(yGrid, centerY), cosAngle)
  );

  // Определение маски
  const mask = tf.logicalAnd(
    tf.less(tf.abs(xRot), width / 2),
    tf.less(tf.abs(yRot), height / 2)
  );

  // Создание ядра и нормализация
  const kernel = tf.where(mask, tf.onesLike(xRot), tf.zerosLike(xRot));
  const sumKernel = tf.sum(kernel);

  return tf.div(kernel, sumKernel);
}

function findLocalMaxima(arr: tf.Tensor1D, threshold: number = 0.5, minDistance: number = 10): Promise<number[]> {
    return arr.array().then((arrData) => {
      // Находим индексы пиков — элементов, которые больше своих соседей
      const peaks: number[] = [];
      
      for (let i = 1; i < arrData.length - 1; i++) {
        if (arrData[i] > arrData[i - 1] && arrData[i] > arrData[i + 1]) {
          peaks.push(i);
        }
      }
  
      // Фильтруем пики по порогу
      const filteredPeaks = peaks.filter(index => arrData[index] > threshold);
  
      // Сортируем пики по значению (от большего к меньшему)
      filteredPeaks.sort((a, b) => arrData[b] - arrData[a]);
  
      // Фильтруем пики по минимальному расстоянию
      const result: number[] = [];
      for (const peak of filteredPeaks) {
        if (result.every(p => Math.abs(peak - p) >= minDistance)) {
          result.push(peak);
        }
      }
  
      return result;
    });
  }

  async function findFingerEdgesImproved(
    conv: tf.Tensor2D, 
    pointX: number, 
    pointY: number, 
    cosAngle: number, 
    sinAngle: number, 
    offset: number, 
    maxDistance: number
  ): Promise<[[number, number] | null, number]> {
  
    const xCoords = tf.range(offset, maxDistance, 1, 'float32');
    const yCoords = tf.range(offset, maxDistance, 1, 'float32');
  
    const x = xCoords.mul(cosAngle).add(pointX).toInt();
    const y = yCoords.mul(sinAngle).add(pointY).toInt();
  
    const [convHeight, convWidth] = conv.shape;
  
    // Создаем маску, чтобы не выходить за пределы массива
    const maskX = x.greaterEqual(0).logicalAnd(x.less(convWidth));
    const maskY = y.greaterEqual(0).logicalAnd(y.less(convHeight));
    const mask = maskX.logicalAnd(maskY);
  
    const validX = x.mul(mask.cast('int32'));
    const validY = y.mul(mask.cast('int32'));
  
    // Выбираем значения из массива вручную
    const validXArray = validX.arraySync() as number[];
    const validYArray = validY.arraySync() as number[];
    const valuesArray: number[] = [];
  
    for (let i = 0; i < validXArray.length; i++) {
      const xVal = validXArray[i];
      const yVal = validYArray[i];
      // Проверяем, что координаты корректны
      if (xVal >= 0 && xVal < convWidth && yVal >= 0 && yVal < convHeight) {
        // Извлекаем скалярное значение из двумерного массива
        const valueTensor = conv.gather([yVal], 0).gather([xVal], 1);
        const valueArray = valueTensor.arraySync() as number[][];  // Извлекаем двумерный массив
        const value = valueArray[0][0];  // Извлекаем скалярное значение
        valueTensor.dispose();
        valuesArray.push(value);
      }
    }
  
    // Проверяем, что valuesArray является одномерным массивом перед созданием тензора
    if (!Array.isArray(valuesArray) || valuesArray.length === 0) {
      throw new Error('valuesArray is not a valid flat array');
    }
  
    const values = tf.tensor1d(valuesArray);
  
    const peaks = await findLocalMaxima(values, 0.5, 10);
  
    if (peaks.length === 0) {
      return [null, 0];
    }
  
    const centerIndex = Math.floor(valuesArray.length / 2);
    const distancesToCenter = peaks.map(peak => Math.abs(peak - centerIndex));
    const sortedIndices = distancesToCenter.map((_, i) => i).sort((a, b) => distancesToCenter[a] - distancesToCenter[b]);
  
    let nearestPeak = peaks[sortedIndices[0]];
    let nearestValue = valuesArray[nearestPeak];
  
    if (peaks.length > 1) {
      const secondNearestPeak = peaks[sortedIndices[1]];
      const secondNearestValue = valuesArray[secondNearestPeak];
  
      if (Math.abs(nearestValue - secondNearestValue) / nearestValue < 0.10) {
        if (nearestPeak > secondNearestPeak) {
          nearestPeak = secondNearestPeak;
          nearestValue = secondNearestValue;
        }
      }
    }
  
    const peakX = validXArray[nearestPeak];
    const peakY = validYArray[nearestPeak];
  
    return [[peakX, peakY], nearestValue];
  }
  
  
  

  async function findFingerWidthImproved(
    prediction: tf.Tensor4D,  // Предсказание 4D: [1, height, width, 1]
    pointX: number,
    pointY: number,
    angleDeg: number,
    kernelHeight: number = 15,
    kernelWidth: number = 7,
    offset: number = 30,
    maxDistance: number = 70
  ): Promise<[([number, number] | null), ([number, number] | null), number, number]> {
  
    const perpAngle = angleDeg + 90;
    const cosPerp = Math.cos(perpAngle * (Math.PI / 180));
    const sinPerp = Math.sin(perpAngle * (Math.PI / 180));
  
    // Создание ядер для свёртки
    const leftKernel = createRectangularKernel(kernelHeight, kernelWidth, perpAngle);
    const rightKernel = leftKernel.reverse(); // Отражаем ядро по обеим осям
  
    // Поскольку prediction имеет форму [1, height, width, 1], просто убираем размерности батча и каналов
    const squeezedPrediction = prediction.squeeze([0, 3]) as tf.Tensor2D;
  
    // Применение свёртки к изображению (восстанавливаем обратно в 4D для свёртки)
    const convLeft = tf.conv2d(
      squeezedPrediction.expandDims(0).expandDims(-1) as tf.Tensor4D,  // Преобразуем обратно в 4D для свёртки
      leftKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D,
      1,  // Шаг свёртки
      'same'
    ).squeeze() as tf.Tensor2D;
  
    const convRight = tf.conv2d(
      squeezedPrediction.expandDims(0).expandDims(-1) as tf.Tensor4D,
      rightKernel.expandDims(-1).expandDims(-1) as tf.Tensor4D,
      1,
      'same'
    ).squeeze() as tf.Tensor2D;
  
    // Поиск границ пальцев
    const [leftEdge, leftMax] = await findFingerEdgesImproved(convLeft, pointX, pointY, -cosPerp, -sinPerp, offset, maxDistance);
    const [rightEdge, rightMax] = await findFingerEdgesImproved(convRight, pointX, pointY, cosPerp, sinPerp, offset, maxDistance);
  
    return [leftEdge, rightEdge, leftMax, rightMax];
  }
  

  function drawSearchRange(
    ctx: CanvasRenderingContext2D,
    centerX: number,
    centerY: number,
    angleDeg: number,
    maxDistance: number,
    kernelHeight: number,
    offset: number
  ) {
    const angleRad = (angleDeg * Math.PI) / 180; // Перевод угла в радианы
    const cosAngle = Math.cos(angleRad);
    const sinAngle = Math.sin(angleRad);
  
    const perpAngle = angleRad + Math.PI / 2;
    const cosPerp = Math.cos(perpAngle);
    const sinPerp = Math.sin(perpAngle);
  
    // 1. Нарисовать линию поиска (желтая линия)
    const leftSearchEnd = [
      Math.floor(centerX - maxDistance * cosPerp),
      Math.floor(centerY - maxDistance * sinPerp),
    ];
    const rightSearchEnd = [
      Math.floor(centerX + maxDistance * cosPerp),
      Math.floor(centerY + maxDistance * sinPerp),
    ];
  
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(leftSearchEnd[0], leftSearchEnd[1]);
    ctx.lineTo(rightSearchEnd[0], rightSearchEnd[1]);
    ctx.stroke();
  
    // 2. Нарисовать два прямоугольника слева и справа от точки поиска
    const boxWidth = maxDistance - offset;
    const halfKernelHeight = Math.floor(kernelHeight / 2);
  
    // Левый прямоугольник
    const leftTopLeft = [
      Math.floor(centerX - offset * cosPerp - halfKernelHeight * cosAngle),
      Math.floor(centerY - offset * sinPerp - halfKernelHeight * sinAngle),
    ];
    const leftTopRight = [
      Math.floor(leftTopLeft[0] - boxWidth * cosPerp),
      Math.floor(leftTopLeft[1] - boxWidth * sinPerp),
    ];
    const leftBottomRight = [
      Math.floor(leftTopRight[0] + kernelHeight * cosAngle),
      Math.floor(leftTopRight[1] + kernelHeight * sinAngle),
    ];
    const leftBottomLeft = [
      Math.floor(leftTopLeft[0] + kernelHeight * cosAngle),
      Math.floor(leftTopLeft[1] + kernelHeight * sinAngle),
    ];
  
    // Правый прямоугольник
    const rightTopRight = [
      Math.floor(centerX + offset * cosPerp - halfKernelHeight * cosAngle),
      Math.floor(centerY + offset * sinPerp - halfKernelHeight * sinAngle),
    ];
    const rightTopLeft = [
      Math.floor(rightTopRight[0] + boxWidth * cosPerp),
      Math.floor(rightTopRight[1] + boxWidth * sinPerp),
    ];
    const rightBottomLeft = [
      Math.floor(rightTopLeft[0] + kernelHeight * cosAngle),
      Math.floor(rightTopLeft[1] + kernelHeight * sinAngle),
    ];
    const rightBottomRight = [
      Math.floor(rightTopRight[0] + kernelHeight * cosAngle),
      Math.floor(rightTopRight[1] + kernelHeight * sinAngle),
    ];
  
    // Рисуем левый прямоугольник (зеленый)
    ctx.strokeStyle = 'green';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(leftTopLeft[0], leftTopLeft[1]);
    ctx.lineTo(leftTopRight[0], leftTopRight[1]);
    ctx.lineTo(leftBottomRight[0], leftBottomRight[1]);
    ctx.lineTo(leftBottomLeft[0], leftBottomLeft[1]);
    ctx.lineTo(leftTopLeft[0], leftTopLeft[1]);
    ctx.stroke();
  
    // Рисуем правый прямоугольник (зеленый)
    ctx.beginPath();
    ctx.moveTo(rightTopLeft[0], rightTopLeft[1]);
    ctx.lineTo(rightTopRight[0], rightTopRight[1]);
    ctx.lineTo(rightBottomRight[0], rightBottomRight[1]);
    ctx.lineTo(rightBottomLeft[0], rightBottomLeft[1]);
    ctx.lineTo(rightTopLeft[0], rightTopLeft[1]);
    ctx.stroke();
  }
  

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
    return tensor
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

// Load the TensorFlow.js model
async function loadModel(): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel('Daniils_sept_6/model.json');
    console.log('Model loaded successfully.');
    return model;
}

// Display results and overlay on the canvas
function postprocessAndDisplay(
    prediction: tf.Tensor4D,
    canvasId: string,
    overlayCanvasId: string,
    centerX: number,
    centerY: number,
    angleDeg: number,
    leftEdge: [number, number] | null,
    rightEdge: [number, number] | null
): void {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    const overlayCanvas = document.getElementById(overlayCanvasId) as HTMLCanvasElement;
    const overlayCtx = overlayCanvas.getContext('2d');

    if (ctx && overlayCtx) {
        prediction.squeeze().array().then((values) => {
            // Проверка типа данных
            if (Array.isArray(values) && Array.isArray(values[0]) && Array.isArray(values[0][0])) {
                console.error("Prediction is a 3D or higher array, but 2D was expected");
                return; // Ничего не делаем, если это 3D или выше
            }

            if (Array.isArray(values) && Array.isArray(values[0])) {
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                const values2D = values as number[][]; // Преобразуем к двумерному массиву

                // Преобразуем двумерный массив предсказаний в формат, подходящий для изображения
                for (let i = 0; i < values2D.length; i++) {
                    for (let j = 0; j < values2D[i].length; j++) {
                        const val = Math.floor(values2D[i][j] * 255);  // Нормализуем значение для RGB
                        const index = (i * canvas.width + j) * 4;
                        data[index] = val;      // Red channel
                        data[index + 1] = val;  // Green channel
                        data[index + 2] = val;  // Blue channel
                        data[index + 3] = 255;  // Alpha channel
                    }
                }

                // Обновляем данные на canvas
                ctx.putImageData(imageData, 0, 0);

                // Отображаем центральную точку
                overlayCtx.fillStyle = 'green';
                overlayCtx.beginPath();
                overlayCtx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
                overlayCtx.fill();

                // Отображаем диапазон поиска
                const maxDistance = 80;
                const kernelHeight = maxDistance * 2;
                const offset = Math.floor(maxDistance / 3);
                drawSearchRange(overlayCtx, centerX, centerY, angleDeg, maxDistance, kernelHeight, offset);

                // Рисуем найденные края пальца
                if (leftEdge && rightEdge) {
                    overlayCtx.strokeStyle = 'red';
                    overlayCtx.lineWidth = 5;
                    overlayCtx.beginPath();
                    overlayCtx.moveTo(leftEdge[0], leftEdge[1]);
                    overlayCtx.lineTo(rightEdge[0], rightEdge[1]);
                    overlayCtx.stroke();
                }
            } else {
                console.error("Unexpected array structure from prediction.");
            }
        });
    }
}





// Main function to execute the logic
async function run() {
    const model = await loadModel();
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

        // Используем await перед вызовом findFingerWidthImproved, так как это асинхронная функция
        for (const [pointX, pointY] of points) {
            const [leftEdge, rightEdge] = await findFingerWidthImproved(normalizedPrediction, pointX, pointY, angleDeg);
            postprocessAndDisplay(normalizedPrediction, 'imageCanvas', 'overlayCanvas', pointX, pointY, angleDeg, leftEdge, rightEdge);
        }
    };
}

run().catch(console.error);
