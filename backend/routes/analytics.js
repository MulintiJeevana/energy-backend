import express from "express";
import { requireAuth } from "../middleware/auth.js";
import { getDailyUsage, getDeviceUsage } from "../controllers/analyticsController.js";

const router = express.Router();

router.use(requireAuth);

router.get("/daily-usage", getDailyUsage);
router.get("/device-usage", getDeviceUsage);

export default router;
