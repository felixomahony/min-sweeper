//import Phaser
import Phaser from "phaser";

// Global variables
//set up the renderer
const height = 512;
const width = 900;
const numClass = 20;
const skyblue = 0x87ceeb;
const lightpink = 0xffb6c1;
const pastelorange = 0xffb347;
const pastelorangeHex = "#ffb347";
const lightgray = 0xf4f4f4;
const darkgray = 0xe1e1e1;
const titleSize = 20;
const textSize = 40;
const textYOffset = 10;

//data view - where the neural network output is displayed
const dViewWidth = 400;
const dViewHeight = 400;
const dViewX = 500;
const dViewY = 105;
const dViewGridCellSize = 5;

//selection view - where the user selects the network inputs
const sViewWidth = 400;
const sViewHeight = 400;
const sViewX = 0;
const sViewY = 105;
const gridCellSize = 50;
const dashSize = 5;
const sViewGridCellSize = 50;
const sViewMinorGridCellSize = 50;

//user inputs
var sViewXRel = 0;
var sViewYRel = 0;

//score
var total_score = 0;
var computer_score = 0;

// Global functions
//neural network functions
function ReLu(x) {
  //Apply ReLu function to input x
  return Math.max(0, x);
}

function sigmoid(x) {
  //Apply Sigmoid function to input x
  return 1 / (1 + Math.exp(-x));
}

function neuralNetwork(context, input_x, input_y) {
  const params = context.params;
  //Apply Neural Network Function to inputs x and y
  // feature map
  const x_0 = input_x;
  const x_1 = input_y;
  const x_2 = input_x * input_x;
  const x_3 = input_y * input_y;

  // first layer
  w_00 = params["weights"]["input_layer"];
  b_00 = params["biases"]["input_layer"];
  z_00 =
    w_00[0][0] * x_0 +
    w_00[0][1] * x_1 +
    w_00[0][2] * x_2 +
    w_00[0][3] * x_3 +
    b_00[0];
  a_00 = Math.max(0, z_00);
  z_01 =
    w_00[1][0] * x_0 +
    w_00[1][1] * x_1 +
    w_00[1][2] * x_2 +
    w_00[1][3] * x_3 +
    b_00[1];
  a_01 = Math.max(0, z_01);
  z_02 =
    w_00[2][0] * x_0 +
    w_00[2][1] * x_1 +
    w_00[2][2] * x_2 +
    w_00[2][3] * x_3 +
    b_00[2];
  a_02 = Math.max(0, z_02);
  z_03 =
    w_00[3][0] * x_0 +
    w_00[3][1] * x_1 +
    w_00[3][2] * x_2 +
    w_00[3][3] * x_3 +
    b_00[3];
  a_03 = Math.max(0, z_03);

  // second layer
  w_01 = params["weights"]["hidden_layer_1"];
  b_01 = params["biases"]["hidden_layer_1"];
  z_10 =
    w_01[0][0] * a_00 +
    w_01[0][1] * a_01 +
    w_01[0][2] * a_02 +
    w_01[0][3] * a_03 +
    b_01[0];
  a_10 = Math.max(0, z_10);
  z_11 =
    w_01[1][0] * a_00 +
    w_01[1][1] * a_01 +
    w_01[1][2] * a_02 +
    w_01[1][3] * a_03 +
    b_01[1];
  a_11 = Math.max(0, z_11);
  z_12 =
    w_01[2][0] * a_00 +
    w_01[2][1] * a_01 +
    w_01[2][2] * a_02 +
    w_01[2][3] * a_03 +
    b_01[2];
  a_12 = Math.max(0, z_12);
  z_13 =
    w_01[3][0] * a_00 +
    w_01[3][1] * a_01 +
    w_01[3][2] * a_02 +
    w_01[3][3] * a_03 +
    b_01[3];
  a_13 = Math.max(0, z_13);

  // third layer
  w_02 = params["weights"]["hidden_layer_2"];
  b_02 = params["biases"]["hidden_layer_2"];
  z_20 =
    w_02[0][0] * a_10 +
    w_02[0][1] * a_11 +
    w_02[0][2] * a_12 +
    w_02[0][3] * a_13 +
    b_02[0];
  a_20 = Math.max(0, z_20);
  z_21 =
    w_02[1][0] * a_10 +
    w_02[1][1] * a_11 +
    w_02[1][2] * a_12 +
    w_02[1][3] * a_13 +
    b_02[1];
  a_21 = Math.max(0, z_21);

  // output layer
  w_03 = params["weights"]["output_layer"];
  b_03 = params["biases"]["output_layer"];

  //now we introduce game inputs
  w_0 = w_03[0][0] + (sViewXRel + context.randomAddition) * 10;
  w_1 = w_03[0][1] + (sViewYRel + context.randomAddition) * 10;

  y_hat = w_0 * a_20 + w_1 * a_21 + b_03[0];
  y_hat = sigmoid(y_hat);

  return { y_hat: y_hat, a: a_20, b: a_21 };
}

function gradient(context) {
  const params = context.params;
  const data = context.data;
  // We use an L2 loss function (L_2 = \Sum(y - y_hat)^2)
  dLdp1 = 0;
  dLdp2 = 0;
  for (let i = 0; i < data["class_0"].length; i++) {
    const x = data["class_0"][i][0];
    const y = data["class_0"][i][1];
    const retVal = neuralNetwork(context, x, y);
    const y_hat = retVal.y_hat;
    const a = retVal.a;
    const b = retVal.b;
    const dLdyhat = 2 * (0 - y_hat) * y_hat * (1 - y_hat);
    dLdp1 += dLdyhat * a;
    dLdp2 += dLdyhat * b;
  }
  for (let i = 0; i < data["class_1"].length; i++) {
    const x = data["class_1"][i][0];
    const y = data["class_1"][i][1];
    const retVal = neuralNetwork(context, x, y);
    const y_hat = retVal.y_hat;
    const a = retVal.a;
    const b = retVal.b;
    const dLdyhat = 2 * (1 - y_hat) * y_hat * (1 - y_hat);
    dLdp1 += dLdyhat * a;
    dLdp2 += dLdyhat * b;
  }

  const normal = Math.sqrt(dLdp1 * dLdp1 + dLdp2 * dLdp2);

  if (normal == 0) {
    return { dLdp1: 0, dLdp2: 0, dLdp1_norm: 0, dLdp2_norm: 0 };
  }

  const dLdp1_norm = dLdp1 / normal;
  const dLdp2_norm = dLdp2 / normal;

  return {
    dLdp1: dLdp1,
    dLdp2: dLdp2,
    dLdp1_norm: dLdp1_norm,
    dLdp2_norm: dLdp2_norm,
  };
}

function score(context) {
  //calculate the score of the current network
  //score is the classification accuracy of the network

  //get the current network
  const params = context.params;
  const data = context.data;

  const correct_0 = data["class_0"].map((coordinates, index) => {
    const retVal = neuralNetwork(context, coordinates[0], coordinates[1]);
    return retVal.y_hat < 0.5;
  });

  const correct_1 = data["class_1"].map((coordinates, index) => {
    const retVal = neuralNetwork(context, coordinates[0], coordinates[1]);
    return retVal.y_hat > 0.5;
  });

  // update objects in context.class_1 to be black if incorrect
  for (let i = 0; i < correct_1.length; i++) {
    if (correct_1[i] == false) {
      context.class_1[i].setStrokeStyle(2, 0x000, 1);
    } else {
      context.class_1[i].setStrokeStyle(0, 0x000, 1);
    }
  }
  // update objects in context.class_0 to be black if incorrect
  for (let i = 0; i < correct_0.length; i++) {
    if (correct_0[i] == false) {
      context.class_0[i].setStrokeStyle(2, 0x000, 1);
    } else {
      context.class_0[i].setStrokeStyle(0, 0x000, 1);
    }
  }

  const correct = correct_0.concat(correct_1);
  const sc = correct.reduce((a, b) => a + b, 0) / correct.length;

  return sc;
}

//drawing functions
function drawBorderDashes(context) {
  //draw background rectangle
  context.add
    .rectangle(
      sViewX + sViewWidth / 2,
      sViewY + sViewHeight / 2,
      sViewWidth,
      sViewHeight,
      lightgray
    )
    .setDepth(-2);
  context.add
    .rectangle(
      sViewX + sViewWidth / 2,
      sViewY + sViewHeight / 2 + 7,
      sViewWidth,
      sViewHeight,
      darkgray
    )
    .setDepth(-3);

  //draw gridlines
  for (let i = sViewX; i <= sViewX + sViewWidth; i += sViewGridCellSize) {
    context.add
      .rectangle(i, sViewY + sViewHeight / 2, 1, sViewHeight, 0xffffff)
      .setDepth(-1);
  }
  for (let i = sViewY; i <= sViewY + sViewHeight; i += sViewGridCellSize) {
    context.add
      .rectangle(sViewX + sViewWidth / 2, i, sViewWidth, 1, 0xffffff)
      .setDepth(-1);
  }

  //draw dashes along the lower border
  for (let i = sViewX; i <= sViewX + sViewWidth - 1; i += sViewGridCellSize) {
    context.add
      .rectangle(i, sViewY + sViewHeight - 7, 1, 14, 0x000)
      .setDepth(-1);
  }
  //draw dashes along the left border
  for (
    let i = sViewY + sViewGridCellSize;
    i <= sViewY + sViewHeight;
    i += sViewGridCellSize
  ) {
    context.add.rectangle(sViewX + 7, i, 14, 1, 0x000).setDepth(-1);
  }
  //draw minor dashes along lower border
  for (
    let i = sViewX;
    i <= sViewX + sViewWidth;
    i += sViewMinorGridCellSize / 5
  ) {
    context.add.rectangle(i, sViewY + sViewHeight - 4, 1, 8, 0x000).setDepth(0);
  }
  //draw minor dashes along left border
  for (
    let i = sViewY;
    i <= sViewY + sViewHeight;
    i += sViewMinorGridCellSize / 5
  ) {
    context.add.rectangle(sViewX + 4, i, 8, 1, 0x000).setDepth(0);
  }

  //draw arrow to the right of left border
  context.add
    .triangle(
      sViewX + 10,
      sViewY + 10,
      sViewX + 10,
      sViewY + 20,
      sViewX + 20,
      sViewY + 10,
      0x000
    )
    .setDepth(0);

  //draw arrow above lower border
  const arrowGraphics = context.add.graphics();
  arrowGraphics.lineStyle(1, 0x000, 1);
  arrowGraphics.beginPath();
  arrowGraphics.moveTo(sViewX + sViewWidth - 20, sViewY + sViewHeight - 15);
  arrowGraphics.lineTo(sViewX + sViewWidth - 10, sViewY + sViewHeight - 20);
  arrowGraphics.lineTo(sViewX + sViewWidth - 20, sViewY + sViewHeight - 25);
  arrowGraphics.moveTo(sViewX + sViewWidth - 10, sViewY + sViewHeight - 20);
  arrowGraphics.lineTo(sViewX + sViewWidth - 60, sViewY + sViewHeight - 20);
  arrowGraphics.strokePath();

  //draw arrow to the right of left border
  const arrowGraphics2 = context.add.graphics();
  arrowGraphics2.lineStyle(1, 0x000, 1);
  arrowGraphics2.beginPath();
  arrowGraphics2.moveTo(sViewX + 15, sViewY + 20);
  arrowGraphics2.lineTo(sViewX + 20, sViewY + 10);
  arrowGraphics2.lineTo(sViewX + 25, sViewY + 20);
  arrowGraphics2.moveTo(sViewX + 20, sViewY + 10);
  arrowGraphics2.lineTo(sViewX + 20, sViewY + 60);
  arrowGraphics2.strokePath();

  //draw text above first arrow
  context.add
    .text(sViewX + sViewWidth - 60, sViewY + sViewHeight - 40, "x", {
      fontFamily: "Arial",
      fontSize: 16,
      color: "#000000",
    })
    .setDepth(0);
  //draw text above second arrow
  context.add
    .text(sViewX + 30, sViewY + 45, "y", {
      fontFamily: "Arial",
      fontSize: 20,
      color: "#000000",
    })
    .setDepth(0);
}

function drawBoundingBox(context) {
  // Draw a background rectangle
  context.add
    .rectangle(
      dViewX + dViewWidth / 2,
      dViewY + dViewHeight / 2,
      dViewWidth,
      dViewHeight,
      0xffffff
    )
    .setDepth(-2);
  context.add
    .rectangle(
      dViewX + dViewWidth / 2,
      dViewY + dViewHeight / 2 + 7,
      dViewWidth,
      dViewHeight,
      darkgray
    )
    .setDepth(-3);
}

function drawBackground(context) {
  context.backgroundGraphics.clear();
  // Draw background based on neural function
  for (let i = dViewX; i <= dViewX + dViewWidth; i += dViewGridCellSize) {
    for (let j = dViewY; j <= dViewY + dViewHeight; j += dViewGridCellSize) {
      const x = ((i - dViewX) / dViewWidth) * 2 - 1;
      const y = ((j - dViewY) / dViewHeight) * 2 - 1;
      const retVal = neuralNetwork(context, x, y);
      const z = retVal.y_hat;
      var color = Phaser.Display.Color.ValueToColor(skyblue);
      if (z > 0.3 && z < 0.5) {
        color = Phaser.Display.Color.Interpolate.ColorWithColor(
          Phaser.Display.Color.ValueToColor(skyblue),
          Phaser.Display.Color.ValueToColor(0xffffff),
          100,
          100 * (z - 0.3) * 5
        );
      } else if (z > 0.5 && z < 0.7) {
        color = Phaser.Display.Color.Interpolate.ColorWithColor(
          Phaser.Display.Color.ValueToColor(0xffffff),
          Phaser.Display.Color.ValueToColor(lightpink),
          100,
          100 * (z - 0.5) * 5
        );
      } else if (z > 0.7) {
        color = Phaser.Display.Color.ValueToColor(lightpink);
      }
      // const color = Phaser.Display.Color.Interpolate.ColorWithColor(
      //   Phaser.Display.Color.ValueToColor(skyblue),
      //   Phaser.Display.Color.ValueToColor(lightpink),
      //   100,
      //   100 * z
      // );
      // const color =
      // z > 0.5
      //   ? Phaser.Display.Color.ValueToColor(lightpink)
      //   : Phaser.Display.Color.ValueToColor(skyblue);
      // console.log(color)
      const colorStyle = Phaser.Display.Color.GetColor(
        color.r,
        color.g,
        color.b
      );
      // graphics.fillStyle(color.color);
      context.backgroundGraphics.fillStyle(colorStyle, 0.3);
      context.backgroundGraphics.fillRect(
        i,
        j,
        dViewGridCellSize,
        dViewGridCellSize
      );
    }
  }
}

//level functions
function preloadLevel(context) {
  // Load image
  context.load.image("next", "assets/next.png");
  context.load.image("next_depressed", "assets/next_depressed.png");
  context.load.image("scoreBg", "assets/scoreBg.png");
  context.load.image("yourScore", "assets/yourScore.png");
  context.load.image("compScore", "assets/compScore.png");
}

function createLevel(context, level) {
  // Draw the selection view
  //draw a grid
  drawBorderDashes(context);

  // Draw the datapoints
  //draw a bounding box
  drawBoundingBox(context);
  //draw background
  context.backgroundGraphics = context.add.graphics().setDepth(-1);
  drawBackground(context);

  context.userUpdate = true;
  //set random number to add to the true zero
  context.randomAddition = Math.random() - 0.5;

  //add circles for each input datum
  context.class_0 = context.data["class_0"].map((coordinates, index) => {
    return context.add.circle(
      (coordinates[0] / 2 + 0.5) * dViewWidth + dViewX,
      (coordinates[1] / 2 + 0.5) * dViewHeight + dViewY,
      5,
      skyblue
    );
  });
  context.class_1 = context.data["class_1"].map((coordinates, index) => {
    return context.add.circle(
      (coordinates[0] / 2 + 0.5) * dViewWidth + dViewX,
      (coordinates[1] / 2 + 0.5) * dViewHeight + dViewY,
      5,
      lightpink
    );
  });

  //add sprite lines to move with selection view
  context.spriteLine1 = context.add
    .line(0, 0, 0, 0, 0, 0, darkgray)
    .setOrigin(0, 0);
  context.spriteLine2 = context.add
    .line(0, 0, 0, 0, 0, 0, darkgray)
    .setOrigin(0, 0);
  context.spriteCircle1 = context.add
    .circle(-100, -100, 5, pastelorange)
    .setOrigin(0, 0);
  context.circle2 = context.add
    .circle(-10, -10, 5, 0xffffff)
    .setOrigin(0, 0)
    .setStrokeStyle(2, pastelorange);

  context.traceLine = context.add.graphics().setDepth(-1);
  context.traceLinePts = [];

  //add image hidden
  context.next = context.add
    .image(-200, 200, "next")
    .setOrigin(0, 0)
    .setDepth(-1);
  context.next_depressed = context.add
    .image(-200, 200, "next_depressed")
    .setOrigin(0, 0)
    .setDepth(-1);

  //add level background
  context.levelBg = context.add
    .image(0, 0, "scoreBg")
    .setOrigin(0, 0)
    .setDepth(-1);
  context.levelText = context.add.text(95, 0, "Level", {
    fontFamily: "DIN Condensed",
    fontSize: titleSize,
    color: "#000000",
  });
  const levelTextSize = context.levelText.displayWidth;
  context.levelText.setPosition(40 - levelTextSize / 2, textYOffset);

  context.level = context.add.text(95, 0, level + 1 + "/5", {
    fontFamily: "DIN Condensed",
    fontSize: textSize,
    color: "#000000",
  });
  const levelSize = context.level.displayWidth;
  context.level.setPosition(40 - levelSize / 2, textYOffset + titleSize);

  // add the accuracy
  context.accuracyBg = context.add
    .image(dViewX + dViewWidth - 80, 0, "compScore")
    .setOrigin(0, 0)
    .setDepth(-1);
  context.accuracyText = context.add.text(95, 0, "You", {
    fontFamily: "DIN Condensed",
    fontSize: titleSize,
    color: "#000000",
  });
  const accuracyTextSize = context.accuracyText.displayWidth;
  context.accuracyText.setPosition(
    dViewX + dViewWidth - 40 - accuracyTextSize / 2,
    textYOffset
  );

  context.accuracy = context.add.text(95, 0, parseInt(total_score * 100), {
    fontFamily: "DIN Condensed",
    fontSize: textSize,
    color: "#000000",
  });

  const accuracySize = context.accuracy.displayWidth;
  context.accuracy.setPosition(
    dViewX + dViewWidth - 40 - accuracySize / 2,
    textYOffset + titleSize
  );

  // add computer accuracy
  context.computerAccuracyBg = context.add
    .image(dViewX + dViewWidth - 80 - 15 - 80, 0, "compScore")
    .setOrigin(0, 0)
    .setDepth(-1);
  context.computerAccuracyText = context.add.text(95, 0, "Computer", {
    fontFamily: "DIN Condensed",
    fontSize: titleSize,
    color: "#000000",
  });
  const computerAccuracyTextSize = context.computerAccuracyText.displayWidth;
  context.computerAccuracyText.setPosition(
    dViewX + dViewWidth - 40 - 80 - 15 - computerAccuracyTextSize / 2,
    textYOffset
  );

  context.computerAccuracy = context.add.text(
    95,
    0,
    parseInt(computer_score * 100),
    {
      fontFamily: "DIN Condensed",
      fontSize: textSize,
      color: "#000000",
    }
  );

  const computerAccuracySize = context.computerAccuracy.displayWidth;
  context.computerAccuracy.setPosition(
    dViewX + dViewWidth - 40 - 80 - 15 - computerAccuracySize / 2,
    textYOffset + titleSize
  );
}

function updateLevel(context, level) {
  // Update game state on each frame
  if (context.userUpdate) {
    // Check if mouse is in the selection view
    if (
      context.input.mousePointer.x > sViewX &&
      context.input.mousePointer.x < sViewX + sViewWidth &&
      context.input.mousePointer.y > sViewY &&
      context.input.mousePointer.y < sViewY + sViewHeight
    ) {
      //move the sprite line to the mouse position
      context.spriteLine1.setTo(
        sViewX + sViewWidth / 2,
        context.input.mousePointer.y,
        context.input.mousePointer.x,
        context.input.mousePointer.y
      );
      context.spriteLine2.setTo(
        context.input.mousePointer.x,
        sViewY + sViewHeight / 2,
        context.input.mousePointer.x,
        context.input.mousePointer.y
      );
      //move the sprite circle to the mouse position
      context.spriteCircle1.setPosition(
        context.input.mousePointer.x - 5,
        context.input.mousePointer.y - 5
      );

      //get the relative position of the mouse in the selection view
      sViewXRel = (context.input.mousePointer.x - sViewX) / sViewWidth;
      sViewXRel = sViewXRel * 2 - 1;
      sViewYRel = (context.input.mousePointer.y - sViewY) / sViewHeight;
      sViewYRel = sViewYRel * 2 - 1;
      drawBackground(context);
    }
  } else {
    //move the sprite line to the mouse position
    if (sViewXRel > 1) {
      sViewXRel = 1;
    }
    if (sViewXRel < -1) {
      sViewXRel = -1;
    }
    if (sViewYRel > 1) {
      sViewYRel = 1;
    }
    if (sViewYRel < -1) {
      sViewYRel = -1;
    }
    const viewportcursorX = ((sViewXRel + 1) / 2) * sViewWidth + sViewX;
    const viewportcursorY = ((sViewYRel + 1) / 2) * sViewHeight + sViewY;

    context.traceLinePts.push({ x: viewportcursorX, y: viewportcursorY });

    const alpha = 0.9;
    const grad = gradient(context);
    context.grad.dLdp1_norm = context.grad.dLdp1_norm * alpha + grad.dLdp1_norm;
    context.grad.dLdp2_norm = context.grad.dLdp2_norm * alpha + grad.dLdp2_norm;

    sViewXRel += context.grad.dLdp1_norm * 0.001;
    sViewYRel += context.grad.dLdp2_norm * 0.001;

    context.spriteLine1.setTo(0, 0, 0, 0);
    context.spriteLine2.setTo(0, 0, 0, 0);

    context.circle2.setPosition(viewportcursorX - 5, viewportcursorY - 5);

    context.traceLine.clear();
    context.traceLine.lineStyle(2, pastelorange);
    context.traceLine.beginPath();
    for (let i = 0; i < context.traceLinePts.length; i++) {
      context.traceLine.lineTo(
        context.traceLinePts[i].x,
        context.traceLinePts[i].y
      );
    }
    context.traceLine.strokePath();

    // console.log("gradient dldp1: " + context.grad.dLdp1);
    // console.log("gradient dldp2: " + context.grad.dLdp2);

    //check if mouse is over the next button
    if (
      context.input.mousePointer.x > width / 2 - 40 &&
      context.input.mousePointer.x < width / 2 + 40 &&
      context.input.mousePointer.y > sViewY + sViewHeight / 2 - 43 &&
      context.input.mousePointer.y < sViewY + sViewHeight / 2 + 43
    ) {
      context.next_depressed.setVisible(true);
      context.next.setVisible(false);
    } else {
      context.next_depressed.setVisible(false);
      context.next.setVisible(true);
    }

    drawBackground(context);
  }
}

function pointerDown(context, level) {
  if (context.userUpdate) {
    if (
      context.input.mousePointer.x > sViewX &&
      context.input.mousePointer.x < sViewX + sViewWidth &&
      context.input.mousePointer.y > sViewY &&
      context.input.mousePointer.y < sViewY + sViewHeight
    ) {
      const sc = score(context);
      console.log("score: " + sc);
      //update accuracy text
      context.accuracy.setText("+" + parseInt(sc * 100));
      //change accuracy text color
      context.accuracy.setColor(pastelorangeHex);
      const accuracySize = context.accuracy.displayWidth;
      context.accuracy.setPosition(
        dViewX + dViewWidth - 40 - accuracySize / 2,
        textYOffset + titleSize
      );

      //update score value
      total_score = total_score + sc;

      context.userUpdate = false;
      //randomise sViewXRel and sViewYRel
      // sViewXRel = Math.random() * 2 - 1;
      // sViewYRel = Math.random() * 2 - 1;
      context.grad = gradient(context);
      //wait 3 seconds
      context.time.delayedCall(
        3000,
        function () {
          context.next.setPosition(
            width / 2 - 40,
            sViewY + sViewHeight / 2 - 43
          );
          context.next_depressed.setPosition(
            width / 2 - 40,
            sViewY + sViewHeight / 2 - 43
          );

          //set computer accuracy
          const sc = score(context);
          computer_score = computer_score + sc;
          console.log("computer score: " + sc);
          //update accuracy text
          context.computerAccuracy.setColor(pastelorangeHex);
          context.computerAccuracy.setText("+" + parseInt(sc * 100));
          //change accuracy text color

          const accuracySize = context.computerAccuracy.displayWidth;
          context.computerAccuracy.setPosition(
            dViewX + dViewWidth - 40 - 80 - 15 - accuracySize / 2,
            textYOffset + titleSize
          );

          //show computer score
          context.computerAccuracyBg.setVisible(true);
          context.computerAccuracyText.setVisible(true);
          context.computerAccuracy.setVisible(true);
        },
        [],
        context
      );
      //hide next_depressed button
      context.next_depressed.setVisible(false);
      console.log("moving to level " + (level + 1));
    }
  } else {
    if (
      context.input.mousePointer.x > width / 2 - 40 &&
      context.input.mousePointer.x < width / 2 + 40 &&
      context.input.mousePointer.y > sViewY + sViewHeight / 2 - 43 &&
      context.input.mousePointer.y < sViewY + sViewHeight / 2 + 43
    ) {
      context.scene.start("level_" + (level + 1));
    }
  }
}

// Levels
//introduction
class Introduction extends Phaser.Scene {
  constructor() {
    super("introduction");
  }

  preload() {
    this.load.image("next", "assets/next.png");
    this.load.image("next_depressed", "assets/next_depressed.png");
  }
  create() {
    //Add welcome text
    this.welcomeText = this.add.text(
      width / 2,
      height / 2 - 100,
      "Welcome to Min Sweeper!",
      {
        fontFamily: "DIN Condensed",
        fontSize: textSize,
        color: "#000",
        align: "center",
      }
    );
    this.welcomeText.setOrigin(0, 0);
    var textWidth = this.welcomeText.displayWidth;
    this.welcomeText.setPosition(width / 2 - textWidth / 2, height / 2 - 100);

    //Add instructions text
    this.instructionsText = this.add.text(
      width / 2,
      height / 2 - 50,
      "Your goal is to choose parameters x and y to best classify the data points shown. \n You will be competing against a computer player using stochastic gradient descent to solve the same problem. \n Good luck!",
      {
        fontFamily: "DIN Condensed",
        fontSize: titleSize,
        color: "#000",
        align: "center",
      }
    );
    this.instructionsText.setOrigin(0, 0);
    textWidth = this.instructionsText.displayWidth;
    this.instructionsText.setPosition(
      width / 2 - textWidth / 2,
      height / 2 - 100 + textSize + 4
    );

    // Add start button (next)
    this.next = this.add.image(width / 2, height / 2 + 100, "next");
    // Add start button (next)
    this.next_depressed = this.add.image(
      width / 2,
      height / 2 + 100,
      "next_depressed"
    );

    // Add onclick
    this.input.on(
      "pointerdown",
      (pointer) => {
        console.log("pointer");
        if (
          this.input.mousePointer.x > width / 2 - 40 &&
          this.input.mousePointer.x < width / 2 + 40 &&
          this.input.mousePointer.y > height / 2 + 100 - 43 &&
          this.input.mousePointer.y < height / 2 + 100 + 43
        ) {
          console.log("starting scene 0");
          this.scene.start("level_0");
        }
      },
      this
    );
  }
  update() {
    if (
      this.input.mousePointer.x > width / 2 - 40 &&
      this.input.mousePointer.x < width / 2 + 40 &&
      this.input.mousePointer.y > height / 2 + 100 - 43 &&
      this.input.mousePointer.y < height / 2 + 100 + 43
    ) {
      this.next_depressed.setVisible(true);
      this.next.setVisible(false);
    } else {
      this.next_depressed.setVisible(false);
      this.next.setVisible(true);
    }
  }
}

//level 0
class Level_0 extends Phaser.Scene {
  constructor() {
    super("level_0");
  }

  preload() {
    this.load.json("data_0", "data/input_data_level_" + 0 + ".json");
    this.load.json("network_params_0", "data/model_level_" + 0 + ".json");
    preloadLevel(this);
    console.log("loaded level " + 0);
  }
  create() {
    //load data from json file
    this.data = this.cache.json.get("data_0");
    this.params = this.cache.json.get("network_params_0");

    createLevel(this, 0);

    //add onclick listener
    this.input.on("pointerdown", (pointer) => {
      pointerDown(this, 0);
    });
  }
  update() {
    updateLevel(this, 0);
  }
}

//level 1
class Level_1 extends Phaser.Scene {
  constructor() {
    super("level_1");
  }

  preload() {
    this.load.json("data_1", "data/input_data_level_" + 1 + ".json");
    this.load.json("network_params_1", "data/model_level_" + 1 + ".json");
    preloadLevel(this);
    console.log("loaded level " + 1);
  }
  create() {
    //load data from json file
    this.data = this.cache.json.get("data_1");
    this.params = this.cache.json.get("network_params_1");

    createLevel(this, 1);

    //add onclick listener
    this.input.on("pointerdown", (pointer) => {
      pointerDown(this, 1);
    });
  }
  update() {
    updateLevel(this, 1);
  }
}

//level 2
class Level_2 extends Phaser.Scene {
  constructor() {
    super("level_2");
  }

  preload() {
    this.load.json("data_2", "data/input_data_level_" + 2 + ".json");
    this.load.json("network_params_2", "data/model_level_" + 2 + ".json");
    preloadLevel(this);
    console.log("loaded level " + 2);
  }
  create() {
    //load data from json file
    this.data = this.cache.json.get("data_2");
    this.params = this.cache.json.get("network_params_2");

    createLevel(this, 2);

    //add onclick listener
    this.input.on("pointerdown", (pointer) => {
      pointerDown(this, 2);
    });
  }
  update() {
    updateLevel(this, 2);
  }
}

//level 3
class Level_3 extends Phaser.Scene {
  constructor() {
    super("level_3");
  }

  preload() {
    this.load.json("data_3", "data/input_data_level_" + 3 + ".json");
    this.load.json("network_params_3", "data/model_level_" + 3 + ".json");
    preloadLevel(this);
    console.log("loaded level " + 3);
  }
  create() {
    //load data from json file
    this.data = this.cache.json.get("data_3");
    this.params = this.cache.json.get("network_params_3");

    createLevel(this, 3);

    //add onclick listener
    this.input.on("pointerdown", (pointer) => {
      pointerDown(this, 3);
    });
  }
  update() {
    updateLevel(this, 3);
  }
}

//level 4
class Level_4 extends Phaser.Scene {
  constructor() {
    super("level_4");
  }

  preload() {
    this.load.json("data_4", "data/input_data_level_" + 4 + ".json");
    this.load.json("network_params_4", "data/model_level_" + 4 + ".json");
    preloadLevel(this);
    console.log("loaded level " + 4);
  }
  create() {
    //load data from json file
    this.data = this.cache.json.get("data_4");
    this.params = this.cache.json.get("network_params_4");

    createLevel(this, 4);

    //add onclick listener
    this.input.on("pointerdown", (pointer) => {
      pointerDown(this, 4);
    });
  }
  update() {
    updateLevel(this, 4);
  }
}

// Levels
//Level 5 (goodbye)
class Level_5 extends Phaser.Scene {
  constructor() {
    super("level_5");
  }

  preload() {
    this.load.image("restart", "assets/restart.png");
    this.load.image("restart_depressed", "assets/restart_depressed.png");
    this.load.image("compScore", "assets/compScore.png");
  }
  create() {
    //Add welcome text
    this.welcomeText = this.add.text(
      width / 2,
      height / 2 - 100,
      total_score > computer_score ? "Congratulations" : "Get 'em next time",
      {
        fontFamily: "DIN Condensed",
        fontSize: textSize,
        color: "#000",
        align: "center",
      }
    );
    this.welcomeText.setOrigin(0, 0);
    var textWidth = this.welcomeText.displayWidth;
    this.welcomeText.setPosition(
      width / 2 - textWidth / 2,
      height / 2 - 100 - 87 - 30
    );

    //Add instructions text
    this.instructionsText = this.add.text(
      width / 2,
      height / 2 - 50,
      total_score > computer_score
        ? "You beat the gradient descent algorithm!\nEver considered a career as a supercomputer?"
        : "Bad luck, you didn't perform as well as the gradient descent algorithm.\nBetter luck next time.",
      {
        fontFamily: "DIN Condensed",
        fontSize: titleSize,
        color: "#000",
        align: "center",
      }
    );
    this.instructionsText.setOrigin(0, 0);
    textWidth = this.instructionsText.displayWidth;
    this.instructionsText.setPosition(
      width / 2 - textWidth / 2,
      height / 2 - 100 + textSize + 4
    );

    // Add start button (next)
    this.next = this.add.image(width / 2, height / 2 + 100, "restart");
    // Add start button (next)
    this.next_depressed = this.add.image(
      width / 2,
      height / 2 + 100,
      "restart_depressed"
    );

    // add the accuracy
    this.accuracyBg = this.add
      .image(width / 2 - 88, height / 2 - 100 - 87 - 15 + textSize, "compScore")
      .setOrigin(0, 0)
      .setDepth(-1);
    this.accuracyText = this.add.text(95, 0, "You", {
      fontFamily: "DIN Condensed",
      fontSize: titleSize,
      color: "#000000",
    });
    const accuracyTextSize = this.accuracyText.displayWidth;
    this.accuracyText.setPosition(
      width / 2 - 48 - accuracyTextSize / 2,
      textYOffset + height / 2 - 100 - 87 - 15 + textSize
    );

    this.accuracy = this.add.text(95, 0, parseInt(total_score * 100), {
      fontFamily: "DIN Condensed",
      fontSize: textSize,
      color: "#000000",
    });

    const accuracySize = this.accuracy.displayWidth;
    this.accuracy.setPosition(
      width / 2 - 48 - accuracySize / 2,
      textYOffset + titleSize + height / 2 - 100 - 87 - 15 + textSize
    );

    // add computer accuracy
    this.computerAccuracyBg = this.add
      .image(width / 2 + 8, height / 2 - 100 - 87 - 15 + textSize, "compScore")
      .setOrigin(0, 0)
      .setDepth(-1);
    this.computerAccuracyText = this.add.text(95, 0, "Computer", {
      fontFamily: "DIN Condensed",
      fontSize: titleSize,
      color: "#000000",
    });
    const computerAccuracyTextSize = this.computerAccuracyText.displayWidth;
    this.computerAccuracyText.setPosition(
      width / 2 + 48 - computerAccuracyTextSize / 2,
      textYOffset + height / 2 - 100 - 87 - 15 + textSize
    );

    this.computerAccuracy = this.add.text(
      95,
      0,
      parseInt(computer_score * 100),
      {
        fontFamily: "DIN Condensed",
        fontSize: textSize,
        color: "#000000",
      }
    );

    const computerAccuracySize = this.computerAccuracy.displayWidth;
    this.computerAccuracy.setPosition(
      width / 2 + 48 - computerAccuracySize / 2,
      textYOffset + titleSize + height / 2 - 100 - 87 - 15 + textSize
    );

    // Add onclick
    this.input.on(
      "pointerdown",
      (pointer) => {
        console.log("pointer");
        if (
          this.input.mousePointer.x > width / 2 - 40 &&
          this.input.mousePointer.x < width / 2 + 40 &&
          this.input.mousePointer.y > height / 2 + 100 - 43 &&
          this.input.mousePointer.y < height / 2 + 100 + 43
        ) {
          console.log("starting introduction");
          total_score = 0;
          computer_score = 0;
          this.scene.start("introduction");
        }
      },
      this
    );
  }
  update() {
    if (
      this.input.mousePointer.x > width / 2 - 40 &&
      this.input.mousePointer.x < width / 2 + 40 &&
      this.input.mousePointer.y > height / 2 + 100 - 43 &&
      this.input.mousePointer.y < height / 2 + 100 + 43
    ) {
      this.next_depressed.setVisible(true);
      this.next.setVisible(false);
    } else {
      this.next_depressed.setVisible(false);
      this.next.setVisible(true);
    }
  }
}

const config = {
  type: Phaser.AUTO,
  width: width,
  height: height,
  backgroundColor: "fff",
  scene: [Introduction, Level_0, Level_1, Level_2, Level_3, Level_4, Level_5],
  antialias: true,
};

const game = new Phaser.Game(config);
