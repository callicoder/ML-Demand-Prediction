var fs = require('fs');
var inputFile = './predictions.json';
var outputFile = '../public/api/hot-zone.json';
var ar = [];
fs.readFile(inputFile, 'utf8', function (err,data) {
  var data = JSON.parse(data);
  data.forEach(r => {
    if (ar[r.hour_of_day] === undefined) {
      ar[r.hour_of_day] = [];
    }
    ar[r.hour_of_day].push({
      geohash: r.geohash,
      weight: r.weight
    });
  });
  fs.writeFile(outputFile, JSON.stringify(ar), ()=> {});
});