"use client";
import React from 'react';
import { Search, Bell, MapPin, ChevronDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
    BarChart, Bar, XAxis, YAxis, ResponsiveContainer,
    LineChart, Line, Tooltip, ComposedChart
} from 'recharts';

// Data mẫu cho biểu đồ
const throughputData = [
    { name: 'Mon', value: 30 },
    { name: 'Tue', value: 45 },
    { name: 'Wed', value: 60 },
    { name: 'Thu', value: 90, highlight: true },
    { name: 'Fri', value: 40 },
    { name: 'Sat', value: 35 },
    { name: 'Sun', value: 25 },
];

const qualityData = [
    { month: 'May', rate: 0.3, line: 0.4 },
    { month: 'Jun', rate: 0.5, line: 0.35 },
    { month: 'Jul', rate: 0.8, line: 0.45 },
    { month: 'Aug', rate: 0.6, line: 0.4 },
    { month: 'Sep', rate: 0.7, line: 0.5 },
];

export default function DashboardPage() {
    return (
        <div className="p-8 bg-gray-50 min-h-screen font-sans">
            {/* Top Header */}
            <header className="flex justify-between items-center mb-8">
                <div className="text-sm text-muted-foreground">
                    Capacity plans / Dashboard / <span className="text-foreground font-semibold">Overview</span>
                </div>
                <div className="flex items-center gap-4">
                    <div className="relative w-64">
                        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input placeholder="Search by serial..." className="pl-8 rounded-full bg-white" />
                    </div>
                    <div className="p-2 bg-white rounded-full border shadow-sm">
                        <MapPin className="h-4 w-4" />
                    </div>
                    <div className="p-2 bg-white rounded-full border shadow-sm">
                        <Bell className="h-4 w-4" />
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-12 gap-6">
                {/* Left Column */}
                <div className="col-span-12 lg:col-span-7 space-y-6">

                    {/* Throughput Card */}
                    <Card className="rounded-3xl border-none shadow-sm overflow-hidden">
                        <CardHeader className="flex flex-row items-center justify-between">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-gray-100 rounded-lg">⇄</div>
                                <CardTitle className="text-3xl font-bold">Throughput</CardTitle>
                            </div>
                            <Badge variant="outline" className="rounded-full px-4 py-1">this week <ChevronDown className="ml-1 h-3 w-3" /></Badge>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="h-[200px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={throughputData}>
                                            <Bar dataKey="value" fill="#E5E7EB" radius={[4, 4, 4, 4]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                    <div className="mt-4">
                                        <span className="text-3xl font-bold">+4.2%</span>
                                        <p className="text-xs text-muted-foreground uppercase">production volume growth</p>
                                    </div>
                                </div>
                                {/* Image Placeholder cho khối 3D */}
                                <div className="flex items-center justify-center bg-slate-50 rounded-2xl">
                                    <div className="text-center text-muted-foreground italic text-sm">
                                        [3D Visualization Canvas]
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    <div className="grid grid-cols-2 gap-6">
                        {/* Worker Capacity */}
                        <Card className="rounded-3xl border-none shadow-sm">
                            <CardHeader className="flex flex-row items-center justify-between">
                                <CardTitle className="text-lg">Worker capacity</CardTitle>
                                <button className="text-xs text-muted-foreground underline">View All</button>
                            </CardHeader>
                            <CardContent className="space-y-6">
                                <WorkerItem name="John S." hours={34} progress={85} avatar="https://github.com/shadcn.png" />
                                <WorkerItem name="Maria J." hours={36.8} progress={92} avatar="https://github.com/shadcn.png" />
                            </CardContent>
                        </Card>

                        {/* Quality Card */}
                        <Card className="rounded-3xl border-none shadow-sm bg-[#D9EAD3]">
                            <CardHeader className="flex flex-row items-center justify-between">
                                <div>
                                    <CardTitle className="text-lg inline mr-2">Quality</CardTitle>
                                    <Badge className="bg-white text-green-700 hover:bg-white text-[10px] h-5">HIGH</Badge>
                                </div>
                                <Badge variant="default" className="text-xs">defect rate <ChevronDown className="ml-1 h-3 w-3" /></Badge>
                            </CardHeader>
                            <CardContent className="h-[150px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={qualityData}>
                                        <Bar dataKey="rate" fill="#B6D7A8" radius={[2, 2, 0, 0]} barSize={20} />
                                        <Line type="monotone" dataKey="line" stroke="#000" dot={false} strokeWidth={1} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>
                    </div>
                </div>

                {/* Right Column */}
                <div className="col-span-12 lg:col-span-5 space-y-6">
                    {/* On-time Delivery */}
                    <Card className="rounded-3xl border-none shadow-sm h-full">
                        <CardHeader className="flex flex-row items-center justify-between">
                            <CardTitle className="text-lg">On-time delivery</CardTitle>
                            <button className="text-xs text-muted-foreground underline">View All</button>
                        </CardHeader>
                        <CardContent className="relative">
                            {/* Timeline Line */}
                            <div className="absolute left-9 top-4 bottom-4 w-px bg-gray-200" />

                            <div className="space-y-8">
                                <DeliveryItem
                                    time="4:00 pm"
                                    title="Urgent Steel Delivery to New York, USA (depo 1)"
                                    status="Dispatch"
                                    pcs="120 pcs"
                                    tons="45 tons"
                                    active
                                />
                                <DeliveryItem
                                    time="6:00 pm"
                                    title="International Freight Shipment to Paris (depo 2)"
                                    status="Preparation"
                                    pcs="1,007 pcs"
                                    tons="213 tons"
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* Total Revenue - Orange Card */}
                    <Card className="rounded-3xl border-none shadow-sm bg-[#FF7A32] text-white overflow-hidden">
                        <CardHeader>
                            <div className="flex justify-between items-center">
                                <div className="flex items-center gap-2">
                                    <div className="p-2 bg-orange-400 rounded-lg">▦</div>
                                    <CardTitle className="text-lg">Total revenue</CardTitle>
                                </div>
                                <Badge variant="outline" className="text-white border-white">USD, $ <ChevronDown className="ml-1 h-3 w-3" /></Badge>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[100px] mb-4">
                                {/* Biểu đồ nến/bar nhỏ màu trắng */}
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={throughputData}>
                                        <Bar dataKey="value" fill="rgba(255,255,255,0.4)" radius={[2, 2, 2, 2]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="flex items-end justify-between">
                                <h2 className="text-4xl font-bold">$2,456,900</h2>
                                <Badge className="bg-black/20 text-white mb-2">+2.5% revenue growth</Badge>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}

// --- Sub-components để code gọn hơn ---

function WorkerItem({ name, hours, progress, avatar }: any) {
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                    <Avatar className="h-8 w-8">
                        <AvatarImage src={avatar} />
                        <AvatarFallback>{name[0]}</AvatarFallback>
                    </Avatar>
                    <span className="font-medium">{name}</span>
                </div>
                <div className="text-muted-foreground flex gap-1 items-center">
                    <span className="text-[10px]">⚒</span> Productivity Hours: <span className="text-foreground font-semibold">{hours}</span>
                </div>
            </div>
            <div className="relative pt-1">
                <Progress value={progress} className="h-1.5" />
                <span className="absolute right-0 -top-1 text-[10px] bg-black text-white px-1.5 rounded-full">{progress}%</span>
            </div>
        </div>
    );
}

function DeliveryItem({ time, title, status, pcs, tons, active = false }: any) {
    return (
        <div className="flex gap-4 relative z-10">
            <div className="text-[10px] text-muted-foreground w-12 pt-1">{time}</div>
            <div className={`flex-1 p-4 rounded-2xl border ${active ? 'bg-white shadow-md' : 'bg-gray-50'}`}>
                <h4 className="text-sm font-semibold mb-2">{title}</h4>
                <p className="text-[10px] text-muted-foreground mb-4">Prioritized to meet critical timelines required by key clients.</p>
                <div className="flex gap-2">
                    <Badge className={active ? "bg-orange-500" : "bg-slate-400"}>{status}</Badge>
                    <Badge variant="outline" className="bg-white">{pcs}</Badge>
                    <Badge variant="outline" className="bg-white">{tons}</Badge>
                </div>
            </div>
        </div>
    );
}