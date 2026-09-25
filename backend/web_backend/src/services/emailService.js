const { Resend } = require("resend");

const resend = new Resend(process.env.RESEND_API_KEY);

const sendOtpEmail = async (email, otp) => {
  const { data, error } = await resend.emails.send({
    from: process.env.EMAIL_FROM,
    to: email,
    subject: "Your Login OTP",
    text: `Your OTP is ${otp}. It is valid for 5 minutes.`,
  });

  if (error) {
    throw new Error(error.message);
  }

  return data;
};

module.exports = {
  sendOtpEmail,
};