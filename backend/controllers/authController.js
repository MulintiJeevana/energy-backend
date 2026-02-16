import bcrypt from 'bcrypt';
import User from '../models/User.js';
import Device from '../models/Device.js';
import EnergyUsage from '../models/EnergyUsage.js';
import { sendOTPEmail } from '../utils/emailService.js';
import { generateOTP } from '../utils/otpGenerator.js';
import { createSampleData } from '../utils/sampleData.js';
import { sendSMS } from '../utils/twilioService.js'

export const signup = async (req, res) => {
  try {
    const { email, phone, password, name } = req.body;

    if (!email || !password || !name) {
      return res.status(400).json({ message: 'All fields are required' });
    }
    console.log("in sgn up")
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 12);

    const otp = generateOTP();
    const otpExpires = new Date(Date.now() + 5 * 60 * 1000); // 5 mins

    const user = new User({
      email,
      phone,
      password: hashedPassword,
      name,
      emailVerificationToken: otp,
      emailVerificationExpires: otpExpires
    });

    await user.save()
    console.log("user saved");

    // await sendOTPEmail(email, otp, name);
    await sendSMS(phone, otp);
    console.log("otp send")
    res.status(201).json({
      message: 'User created successfully. Please check your email for OTP verification.',
      userId: user._id
    });

  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ message: 'Server error during signup' });
  }
};

export const verifyOTP = async (req, res) => {
  try {
    const { email, otp } = req.body;

    if (!email || !otp) {
      return res.status(400).json({ message: "Email and OTP are required" });
    }

    const user = await User.findOne({ email });

    if (!user) {
      return res.status(400).json({ message: "User not found" });
    }

    // If OTP expired → delete user
    if (user.emailVerificationExpires < Date.now()) {
      await User.deleteOne({ email });
      return res.status(400).json({ message: "OTP expired. Please signup again." });
    }

    // If OTP incorrect
    if (user.emailVerificationToken !== otp) {
      return res.status(400).json({ message: "Invalid OTP" });
    }

    // Success case
    user.isEmailVerified = true;
    user.emailVerificationToken = null;
    user.emailVerificationExpires = null;

    await user.save();
    await createSampleData(user._id);

    res.json({ message: "Email verified successfully" });

  } catch (error) {
    console.error("OTP verification error:", error);
    res.status(500).json({ message: "Server error during OTP verification" });
  }
};


export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ message: 'Email and password are required' });
    }

    const user = await User.findOne({ email });
    if (!user || !user.isEmailVerified) {
      return res.status(400).json({ message: 'Invalid credentials or email not verified' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    req.session.userId = user._id;

    req.session.userEmail = user.email;
    console.log("logged in")
    req.session.userName = user.name;
    console.log(req.session)

    req.session.save((err) => {
      if (err) {
        console.error("❌ Session save error:", err);
        return res.status(500).json({ message: "Failed to save session" });
      }

      res.json({
        message: "Login successful",
        user: {
          id: user._id,
          email: user.email,
          name: user.name,
        },
      });
    });

    // res.json({
    //   message: 'Login successful',
    //   user: {
    //     id: user._id,
    //     email: user.email,
    //     name: user.name
    //   }
    // });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error during login' });
  }
};

export const logout = (req, res) => {
  req.session.destroy(err => {
    if (err) {
      return res.status(500).json({ message: 'Could not log out' });
    }

    console.log("in logout")
    res.clearCookie('connect.sid');
    res.json({ message: 'Logged out successfully' });
  });
};

export const getProfile = async (req, res) => {
  try {
    const user = await User.findById(req.session.userId).select('-password');
    console.log("in profile")
    if (!user) {
      return res.status(404).json({ success: true, message: 'User not found' });
    }
    res.status(200).json({ success: true, message: "good profile", user: user });
  } catch (error) {

    console.error('Profile error:', error);
    res.status(500).json({ message: 'Server error during profile fetch' });
  }
};

export const resendOTP = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ message: 'Email is required' });
    }

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    if (user.isEmailVerified) {
      return res.status(400).json({ message: 'Email already verified' });
    }

    const otp = generateOTP();
    const otpExpires = new Date(Date.now() + 5 * 60 * 1000);

    user.emailVerificationToken = otp;
    user.emailVerificationExpires = otpExpires;
    await user.save();

    await sendOTPEmail(email, otp, user.name);

    res.json({ message: 'OTP resent successfully' });
  } catch (error) {
    console.error('Resend OTP error:', error);
    res.status(500).json({ message: 'Server error during OTP resend' });
  }
};
