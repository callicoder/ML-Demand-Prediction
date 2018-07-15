var express = require('express');
var { exec } = require('child_process');
var router = express.Router();

/* GET users listing. */
router.get('/recalculate', function(req, res, next) {
  exec('python recalculate-zones.py');
  res.send('job scheduled');
});

module.exports = router;
