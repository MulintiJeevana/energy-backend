import EnergyUsage from '../models/EnergyUsage.js';
import Device from '../models/Device.js';

export const getEnergyStats = async (req, res) => {
  try {
    const { period = 'week' } = req.query;
    const userId = req.session.userId;

    let dateFilter = {};
    const now = new Date();

    if (period === 'week') {
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      dateFilter = { date: { $gte: weekAgo } };
    } else if (period === 'month') {
      const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      dateFilter = { date: { $gte: monthAgo } };
    }

    // Get total energy consumption
    const totalConsumption = await EnergyUsage.aggregate([
      { $match: { userId: userId, ...dateFilter } },
      { $group: { _id: null, total: { $sum: '$energyConsumed' }, totalCost: { $sum: '$cost' } } }
    ]);

    // Get device-wise consumption
    const deviceConsumption = await EnergyUsage.aggregate([
      { $match: { userId: userId, ...dateFilter } },
      { 
        $group: { 
          _id: '$deviceName', 
          totalEnergy: { $sum: '$energyConsumed' },
          totalCost: { $sum: '$cost' },
          totalHours: { $sum: '$hoursUsed' }
        } 
      },
      { $sort: { totalEnergy: -1 } }
    ]);

    // Get daily consumption for the period
    const dailyConsumption = await EnergyUsage.aggregate([
      { $match: { userId: userId, ...dateFilter } },
      {
        $group: {
          _id: { $dateToString: { format: '%Y-%m-%d', date: '$date' } },
          totalEnergy: { $sum: '$energyConsumed' },
          totalCost: { $sum: '$cost' }
        }
      },
      { $sort: { '_id': 1 } }
    ]);

    // Get recommendations
    const recommendations = await generateRecommendations(userId);

    res.json({
      totalConsumption: totalConsumption[0] || { total: 0, totalCost: 0 },
      deviceConsumption,
      dailyConsumption,
      recommendations
    });
  } catch (error) {
    console.error('Get energy stats error:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

export const addEnergyUsage = async (req, res) => {
  try {
    const { deviceId, hoursUsed, date } = req.body;
    const userId = req.session.userId;

    const device = await Device.findOne({ _id: deviceId, userId });
    if (!device) {
      return res.status(404).json({ message: 'Device not found' });
    }

    const energyConsumed = (device.wattage * hoursUsed) / 1000; // Convert to kWh
    const cost = energyConsumed * 0.12; // Assuming $0.12 per kWh

    const energyUsage = new EnergyUsage({
      userId,
      deviceId,
      deviceName: device.name,
      wattage: device.wattage,
      hoursUsed,
      energyConsumed,
      cost,
      date: date || new Date()
    });

    await energyUsage.save();
    res.status(201).json(energyUsage);
  } catch (error) {
    console.error('Add energy usage error:', error);
    res.status(500).json({ message: 'Server error' });
  }
};

const generateRecommendations = async (userId) => {
  try {
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    
    // Get top energy consuming devices
    const topDevices = await EnergyUsage.aggregate([
      { $match: { userId: userId, date: { $gte: weekAgo } } },
      { 
        $group: { 
          _id: '$deviceName', 
          totalEnergy: { $sum: '$energyConsumed' },
          percentage: { $sum: '$energyConsumed' }
        } 
      },
      { $sort: { totalEnergy: -1 } },
      { $limit: 3 }
    ]);

    // Calculate total consumption for percentage
    const totalConsumption = await EnergyUsage.aggregate([
      { $match: { userId: userId, date: { $gte: weekAgo } } },
      { $group: { _id: null, total: { $sum: '$energyConsumed' } } }
    ]);

    const total = totalConsumption[0]?.total || 1;
    
    const recommendations = [];

    topDevices.forEach((device, index) => {
      const percentage = Math.round((device.totalEnergy / total) * 100);
      
      if (index === 0 && percentage > 40) {
        recommendations.push(`Your ${device._id} is consuming ${percentage}% of your weekly energy. Consider reducing usage during peak hours.`);
      } else if (device._id.toLowerCase().includes('ac') || device._id.toLowerCase().includes('air')) {
        recommendations.push(`Use your ${device._id} less between 3-5PM to save energy and reduce costs.`);
      } else if (device._id.toLowerCase().includes('heater')) {
        recommendations.push(`Your ${device._id} is consuming ${percentage}% of your weekly energy. Consider lowering the temperature by 2°C.`);
      }
    });

    if (recommendations.length === 0) {
      recommendations.push('Great job! Your energy consumption is well-balanced across devices.');
    }

    return recommendations;
  } catch (error) {
    console.error('Generate recommendations error:', error);
    return ['Unable to generate recommendations at this time.'];
  }
};