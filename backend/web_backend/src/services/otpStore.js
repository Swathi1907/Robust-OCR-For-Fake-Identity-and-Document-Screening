const otpStore = new Map();

const saveOtp = (employeeId, otp) => {
  const expiresAt = Date.now() + 5 * 60 * 1000;

  otpStore.set(employeeId, {
    otp,
    expiresAt,
  });
};

const getOtp = (employeeId) => {
  return otpStore.get(employeeId);
};

const deleteOtp = (employeeId) => {
  otpStore.delete(employeeId);
};

module.exports = {
  saveOtp,
  getOtp,
  deleteOtp,
};