const express = require("express");

const router = express.Router();
const {requestOtp} = require('../controllers/authControllers');

router.post("/request-otp",requestOtp);

router.post("/verify-otp", (req, res) => {
  res.json({
    success: true,
    message: "OTP verification request received",
  });
});

module.exports = router;