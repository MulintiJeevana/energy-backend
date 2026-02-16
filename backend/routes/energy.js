import express from 'express';
import { getEnergyStats, addEnergyUsage } from '../controllers/energyController.js';
import { requireAuth } from '../middleware/auth.js';

const router = express.Router();

router.use(requireAuth);

router.get('/stats', getEnergyStats);
router.post('/usage', addEnergyUsage);

export default router;