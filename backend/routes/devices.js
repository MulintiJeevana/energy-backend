import express from 'express';
import { getDevices, createDevice, updateDevice, deleteDevice } from '../controllers/deviceController.js';
import { requireAuth } from '../middleware/auth.js';

const router = express.Router();

console.log("in th routes of the devices")
// router.use(requireAuth);

router.get('/', getDevices);
router.post('/', createDevice);
router.put('/:id', updateDevice);
router.delete('/:id', deleteDevice);

export default router;