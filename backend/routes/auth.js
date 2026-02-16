import express from 'express';
import { signup, login, logout, verifyOTP, getProfile, resendOTP } from '../controllers/authController.js';
import { requireAuth } from '../middleware/auth.js';


const router = express.Router();

router.post('/signup', signup);
router.post('/login', login);
router.post('/logout', logout);
router.post('/verify-otp', verifyOTP);
router.post('/resend-otp', resendOTP);
router.get('/profile', requireAuth, getProfile);

export default router;