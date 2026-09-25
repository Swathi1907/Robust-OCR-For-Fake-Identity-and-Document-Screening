const { generateOtp } = require("../services/otpService");
const { saveOtp } = require("../services/otpStore");
const { sendOtpEmail } = require("../services/emailService");
const User = require("../models/userModel");

const requestOtp = async (req, res) => {
  try {
    const { employeeId } = req.body;

    const user = await User.findOne({
      employeeId,
      isActive: true,
    });

    if (!user) {
      return res.status(401).json({
        success: false,
        message: "Invalid employee ID",
      });
    }

    const otp = generateOtp();

    saveOtp(employeeId, otp);

    await sendOtpEmail(user.email, otp);

    return res.status(200).json({
      success: true,
      message: "OTP sent successfully",
    });
  } catch (error) {
    console.error("Request OTP error:", error);

    return res.status(500).json({
      success: false,
      message: "Unable to send OTP",
    });
  }
};

module.exports = {
  requestOtp,
};