const path = require('path');

module.exports = {
  entry: './game.js', // Path to your main game file
  output: {
    path: path.resolve(__dirname, 'dist'), // Output directory
    filename: 'bundle.js', // Output bundle filename
  },
  module: {
    rules: [
      {
        test: /\.js$/, // Apply the following loaders to JavaScript files
        exclude: /node_modules/,
        use: 'babel-loader', // Use Babel to transpile JavaScript files
      },
      {
        test: /\.json$/,
        loader: 'json-loader',
        type: 'javascript/auto',
      },
    ],
  },
};