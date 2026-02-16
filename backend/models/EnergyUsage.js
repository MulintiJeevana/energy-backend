import mongoose from 'mongoose';

const energyUsageSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  deviceId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Device',
    required: true
  },
  deviceName: {
    type: String,
    required: true
  },
  wattage: {
    type: Number,
    required: true
  },
  hoursUsed: {
    type: Number,
    required: true
  },
  energyConsumed: {
    type: Number,
    required: true // in kWh
  },
  cost: {
    type: Number,
    required: true // in currency
  },
  date: {
    type: Date,
    default: Date.now
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.model('EnergyUsage', energyUsageSchema);