export const requireAuth = (req, res, next) => {
  console.log("session id ", req.session.userId)
  if (!req.session.userId) {
    console.log("in authcheck")
    return res.status(401).json({ message: 'Authentication required' });
  }
  console.log("after the session check")
  next();
};
