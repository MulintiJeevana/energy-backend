import mongoose from 'mongoose';

const deviceSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: {
    type: String,
    required: true
  },
  type: {
    type: String,
    required: true,
    enum: [
      'AC',
      'Fridge',
      'TV',
      'WashingMachine',
      'Geyser',
      'CeilingFan',
      'WiFiRouter',
      'Lights',
      'Other'
    ]
  },
  wattage: {
    type: Number,
    required: true
  },
  location: {
    type: String,
    enum: [
      'Living Room',
      'Bedroom',
      'Kitchen',
      'Bathroom',
      'Balcony',
      'Other'
    ],
    default: 'Living Room'
  },
  isActive: {
    type: Boolean,
    default: true
  },
  dailyUsage: {
    type: Number, // kWh per day (optional metric we can track)
    default: 0
  },
  currentUsage: {
    type: Number, // current consumption in kW at a given time
    default: 0
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.model('Device', deviceSchema);
