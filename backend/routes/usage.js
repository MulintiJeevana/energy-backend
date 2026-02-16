import express from "express";
import { MongoClient } from "mongodb";
import dotenv from "dotenv";
dotenv.config();

const router = express.Router();

// MongoDB connection string (from .env)
const uri = process.env.MONGO_URI;
let client;

// Reuse a single client
async function getClient() {
    if (!client) {
        if (!uri) {
            throw new Error("❌ MONGO_URI is not defined. Check your .env file.");
        }
        client = new MongoClient(uri);
        await client.connect();
        console.log("✅ Connected to MongoDB");
    }
    return client;
}



router.get("/stats", async (req, res) => {
    try {
        const dbClient = await getClient();
        const db = dbClient.db("energyDB");
        const collection = db.collection("energy");

        // ========================
        // 📅 Date Ranges
        // ========================
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const tomorrow = new Date(today);
        tomorrow.setDate(today.getDate() + 1);

        const weekAgo = new Date(today);
        weekAgo.setDate(today.getDate() - 7);

        const monthAgo = new Date(today);
        monthAgo.setMonth(today.getMonth() - 1);

        // ========================
        // 🔎 Aggregations
        // ========================
        const todayUsage = await collection.aggregate([
            { $match: { timestamp: { $gte: today, $lt: tomorrow } } },
            { $group: { _id: null, total: { $sum: "$energy_consumed" } } }
        ]).toArray();

        const weekUsage = await collection.aggregate([
            { $match: { timestamp: { $gte: weekAgo, $lt: tomorrow } } },
            { $group: { _id: null, total: { $sum: "$energy_consumed" } } }
        ]).toArray();

        const monthUsage = await collection.aggregate([
            { $match: { timestamp: { $gte: monthAgo, $lt: tomorrow } } },
            { $group: { _id: null, total: { $sum: "$energy_consumed" } } }
        ]).toArray();

        // ========================
        // 📤 Response
        // ========================
        res.json({
            today: todayUsage[0]?.total || 0,
            week: weekUsage[0]?.total || 0,
            month: monthUsage[0]?.total || 0,
        });

    } catch (err) {
        console.error("❌ Error in /stats route:", err.message);
        res.status(500).json({ error: err.message });
    }
});


// 📊 Hourly Usage
// GET /api/usage/hourly?date=YYYY-MM-DD
router.get("/hourly", async (req, res) => {
    try {
        const client = await getClient();
        const db = client.db("energyDB");
        const collection = db.collection("energy");

        // pick date (or default latest date in DB)
        let queryDate;
        if (req.query.date) {
            queryDate = new Date(req.query.date);
        } else {
            const latestDoc = await collection
                .find({})
                .sort({ date: -1 })
                .limit(1)
                .toArray();
            queryDate = latestDoc[0].date;
        }

        const hourly = await collection
            .aggregate([
                { $match: { date: queryDate } },
                {
                    $group: {
                        _id: "$hour_of_day",
                        usage: { $sum: "$energy_consumed" },
                    },
                },
                { $sort: { _id: 1 } },
                {
                    $project: {
                        hour: "$_id",
                        usage: 1,
                        _id: 0,
                    },
                },
            ])
            .toArray();

        res.json(hourly);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Failed to fetch hourly usage" });
    }
});

// 📊 Daily Usage
router.get("/daily", async (req, res) => {
    try {
        const client = await getClient();
        const db = client.db("energyDB");
        const collection = db.collection("energy");

        const month = parseInt(req.query.month);
        const year = parseInt(req.query.year);

        let matchStage = {};
        if (!isNaN(month) && !isNaN(year)) {
            // Build date range for that month
            const startDate = new Date(year, month - 1, 1);   // e.g., 2024-03-01
            const endDate = new Date(year, month, 1);         // e.g., 2024-04-01
            matchStage = { date: { $gte: startDate, $lt: endDate } };
        }

        const daily = await collection
            .aggregate([
                { $match: matchStage },
                {
                    $group: {
                        _id: {
                            day: { $dayOfMonth: "$date" },
                            month: { $month: "$date" },
                            year: { $year: "$date" },
                        },
                        usage: { $sum: "$energy_consumed" },
                    },
                },
                { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1 } },
                {
                    $project: {
                        date: {
                            $dateToString: {
                                format: "%Y-%m-%d", date: {
                                    $dateFromParts: {
                                        year: "$_id.year",
                                        month: "$_id.month",
                                        day: "$_id.day"
                                    }
                                }
                            }
                        },
                        usage: 1,
                        _id: 0,
                    },
                },
            ])
            .toArray();

        res.json(daily);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Failed to fetch daily usage" });
    }
});

// router.get("/devices", async (req, res) => {
//     try {
//         const client = await getClient();
//         const db = client.db("energyDB");
//         const usageCollection = db.collection("energy");

//         const { date } = req.query;
//         if (!date) {
//             return res.status(400).json({ error: "date is required (YYYY-MM-DD)" });
//         }

//         // Parse input date
//         const parsed = new Date(date + "T00:00:00.000Z"); // force UTC midnight
//         const nextDay = new Date(parsed);
//         nextDay.setUTCDate(nextDay.getUTCDate() + 1);

//         console.log("DEBUG startDate:", parsed, " endDate:", nextDay);

//         // Aggregate usage for that day
//         const usage = await usageCollection.aggregate([
//             {
//                 $match: {
//                     date: { $gte: parsed, $lt: nextDay }
//                 }
//             },
//             {
//                 $group: {
//                     _id: "$device_id",
//                     totalUsage: { $sum: "$energy_consumed" }
//                 }
//             },
//             {
//                 $project: {
//                     _id: 0,
//                     device: "$_id",
//                     totalUsage: 1
//                 }
//             }
//         ]).toArray();

//         res.json(usage);
//     } catch (err) {
//         console.error("Error in /devices:", err);
//         res.status(500).json({ error: "Failed to fetch devices usage" });
//     }
// });


// 📊 Device Breakdown with cumulative slots
// GET /api/usage/devices?date=YYYY-MM-DD
router.get("/devices", async (req, res) => {
    try {
        const client = await getClient();
        const db = client.db("energyDB");
        const collection = db.collection("energy");

        const { date } = req.query;
        if (!date) {
            return res.status(400).json({ error: "date is required (YYYY-MM-DD)" });
        }

        // Parse input date
        const parsed = new Date(date + "T00:00:00.000Z");
        const nextDay = new Date(parsed);
        nextDay.setUTCDate(nextDay.getUTCDate() + 1);

        // Current cutoff
        const now = new Date();
        const isToday = parsed.toDateString() === now.toDateString();

        const matchStage = {
            date: { $gte: parsed, $lt: nextDay },
        };
        if (isToday) {
            // only include up to current time
            const cutoff = now.toISOString().slice(0, 19).replace("T", " ");
            matchStage.timestamp = { $lte: cutoff };
        }

        // Aggregate usage per device per slot (20-min granularity)
        const pipeline = [
            { $match: matchStage },
            {
                $group: {
                    _id: { device: "$device_id", slot: "$timestamp" },
                    cumulative: { $max: "$cumulative_daily_usage_device" },
                },
            },
            { $sort: { "_id.device": 1, "_id.slot": 1 } },
            {
                $group: {
                    _id: "$_id.device",
                    slots: { $push: "$cumulative" },
                    dailyUsage: { $max: "$cumulative" }, // last cumulative = total up to now
                },
            },
            {
                $project: {
                    _id: 0,
                    name: "$_id",
                    slots: 1,
                    dailyUsage: 1,
                    currentUsage: { $arrayElemAt: ["$slots", -1] },
                },
            },
            { $sort: { name: 1 } },
        ];

        const devices = await collection.aggregate(pipeline).toArray();
        res.json(devices);
    } catch (err) {
        console.error("Error in /devices:", err);
        res.status(500).json({ error: "Failed to fetch devices usage" });
    }
});



export default router;
