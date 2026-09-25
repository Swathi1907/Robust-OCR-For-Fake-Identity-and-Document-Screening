const express= require('express')
const app=express();
const cors= require('cors');

app.use(express.json());
app.use(cors());

app.get("/",(req,res)=>{
   res.json({
    success: true,
    message: "Hello World",
   })
})
const authRoutes=require('./routes/authRoutes')
app.use("/api/auth",authRoutes);
module.exports=app;
