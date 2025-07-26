import React, { useState, useEffect } from 'react'
import { Shield, Eye, Lock, AlertTriangle, TrendingUp, BarChart3 } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

const PrivacyDashboard = () => {
  const [privacyMetrics, setPrivacyMetrics] = useState(null)
  const [budgetHistory, setBudgetHistory] = useState([])
  const [institutionBudgets, setInstitutionBudgets] = useState([])

  useEffect(() => {
    // Mock privacy metrics
    setPrivacyMetrics({
      totalEpsilonSpent: 3.2,
      totalEpsilonBudget: 10.0,
      totalDeltaSpent: 1e-5,
      totalDeltaBudget: 1e-4,
      activeInstitutions: 3,
      totalQueries: 47,
      averageNoiseMagnitude: 0.15,
      privacyLossRate: 0.08
    })

    // Mock budget history over time
    setBudgetHistory([
      { time: '00:00', epsilon: 0, delta: 0 },
      { time: '04:00', epsilon: 0.5, delta: 2e-6 },
      { time: '08:00', epsilon: 1.2, delta: 5e-6 },
      { time: '12:00', epsilon: 2.1, delta: 8e-6 },
      { time: '16:00', epsilon: 2.8, delta: 1.2e-5 },
      { time: '20:00', epsilon: 3.2, delta: 1.5e-5 },
    ])

    // Mock institution budget usage
    setInstitutionBudgets([
      { name: 'General Hospital', epsilon: 1.2, delta: 5e-6, queries: 18 },
      { name: 'Medical Clinic', epsilon: 0.8, delta: 3e-6, queries: 12 },
      { name: 'Research Institute', epsilon: 1.2, delta: 7e-6, queries: 17 },
    ])
  }, [])

  const getPrivacyLevel = (spent, budget) => {
    const ratio = spent / budget
    if (ratio < 0.3) return { level: 'High', color: 'text-green-400', bg: 'bg-green-500/20' }
    if (ratio < 0.7) return { level: 'Medium', color: 'text-yellow-400', bg: 'bg-yellow-500/20' }
    return { level: 'Low', color: 'text-red-400', bg: 'bg-red-500/20' }
  }

  if (!privacyMetrics) return <div>Loading...</div>

  const epsilonLevel = getPrivacyLevel(privacyMetrics.totalEpsilonSpent, privacyMetrics.totalEpsilonBudget)
  const deltaLevel = getPrivacyLevel(privacyMetrics.totalDeltaSpent, privacyMetrics.totalDeltaBudget)

  return (
    <div className="text-white">
      <div className="flex items-center mb-6">
        <Shield className="w-8 h-8 mr-3 text-green-400" />
        <h2 className="text-2xl font-bold">Privacy Dashboard</h2>
      </div>

      {/* Privacy Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Eye className="w-6 h-6 text-blue-400" />
            <span className={`px-2 py-1 rounded-full text-xs ${epsilonLevel.bg} ${epsilonLevel.color}`}>
              {epsilonLevel.level}
            </span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {privacyMetrics.totalEpsilonSpent.toFixed(1)}
          </div>
          <div className="text-sm text-white/60">Epsilon Spent</div>
          <div className="text-xs text-white/40 mt-1">
            of {privacyMetrics.totalEpsilonBudget} total budget
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Lock className="w-6 h-6 text-purple-400" />
            <span className={`px-2 py-1 rounded-full text-xs ${deltaLevel.bg} ${deltaLevel.color}`}>
              {deltaLevel.level}
            </span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {privacyMetrics.totalDeltaSpent.toExponential(1)}
          </div>
          <div className="text-sm text-white/60">Delta Spent</div>
          <div className="text-xs text-white/40 mt-1">
            of {privacyMetrics.totalDeltaBudget.toExponential(1)} budget
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <BarChart3 className="w-6 h-6 text-green-400" />
            <TrendingUp className="w-4 h-4 text-green-400" />
          </div>
          <div className="text-2xl font-bold mb-1">
            {privacyMetrics.totalQueries}
          </div>
          <div className="text-sm text-white/60">Total Queries</div>
          <div className="text-xs text-white/40 mt-1">
            {privacyMetrics.activeInstitutions} active institutions
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <AlertTriangle className="w-6 h-6 text-yellow-400" />
            <span className="text-xs text-white/60">Rate</span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {(privacyMetrics.privacyLossRate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-white/60">Privacy Loss Rate</div>
          <div className="text-xs text-white/40 mt-1">
            per hour average
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Privacy Budget Over Time */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Privacy Budget Usage Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={budgetHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="rgba(255,255,255,0.6)" />
                <YAxis stroke="rgba(255,255,255,0.6)" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="epsilon" 
                  stroke="#60a5fa" 
                  strokeWidth={2}
                  name="Epsilon (ε)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Institution Budget Usage */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Budget Usage by Institution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={institutionBudgets}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.6)" />
                <YAxis stroke="rgba(255,255,255,0.6)" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0,0,0,0.8)', 
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="epsilon" fill="#8b5cf6" name="Epsilon Used" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Detailed Privacy Analysis */}
      <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Privacy Composition Analysis</h3>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Epsilon Budget</span>
                <span>{privacyMetrics.totalEpsilonSpent.toFixed(1)} / {privacyMetrics.totalEpsilonBudget}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                  style={{ width: `${(privacyMetrics.totalEpsilonSpent / privacyMetrics.totalEpsilonBudget) * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Delta Budget</span>
                <span>{privacyMetrics.totalDeltaSpent.toExponential(1)} / {privacyMetrics.totalDeltaBudget.toExponential(1)}</span>
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-green-500 to-teal-600 h-2 rounded-full"
                  style={{ width: `${(privacyMetrics.totalDeltaSpent / privacyMetrics.totalDeltaBudget) * 100}%` }}
                />
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h4 className="font-medium mb-3">Composition Bounds</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/60">Basic Composition:</span>
                  <span>ε = {privacyMetrics.totalEpsilonSpent.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">Advanced Composition:</span>
                  <span>ε ≈ {(privacyMetrics.totalEpsilonSpent * 0.8).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">RDP Bound:</span>
                  <span>ε ≈ {(privacyMetrics.totalEpsilonSpent * 0.6).toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Noise Characteristics</h3>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {privacyMetrics.averageNoiseMagnitude.toFixed(3)}
                </div>
                <div className="text-sm text-white/60">Avg Noise Magnitude</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">
                  Gaussian
                </div>
                <div className="text-sm text-white/60">Noise Distribution</div>
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex items-center">
                <Shield className="w-4 h-4 text-green-400 mr-2" />
                <span>Gradient clipping with L2 norm bound</span>
              </div>
              <div className="flex items-center">
                <Lock className="w-4 h-4 text-blue-400 mr-2" />
                <span>Gaussian mechanism for (ε,δ)-DP</span>
              </div>
              <div className="flex items-center">
                <BarChart3 className="w-4 h-4 text-purple-400 mr-2" />
                <span>Moments accountant for tight bounds</span>
              </div>
              <div className="flex items-center">
                <AlertTriangle className="w-4 h-4 text-yellow-400 mr-2" />
                <span>Real-time privacy loss tracking</span>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10">
              <h4 className="font-medium mb-2">Security Guarantees</h4>
              <div className="text-xs text-white/60 space-y-1">
                <p>• Mathematical privacy guarantees via differential privacy</p>
                <p>• Cryptographic integrity with threshold-ECDSA signatures</p>
                <p>• On-chain verification of privacy parameters</p>
                <p>• Automated budget enforcement and tracking</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PrivacyDashboard