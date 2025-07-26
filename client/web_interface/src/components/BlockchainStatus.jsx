import React, { useState, useEffect } from 'react'
import { Database, Zap, Shield, Users, TrendingUp, Clock, CheckCircle, AlertTriangle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'

const BlockchainStatus = () => {
  const [networkStats, setNetworkStats] = useState(null)
  const [canisterMetrics, setCanisterMetrics] = useState([])
  const [transactionHistory, setTransactionHistory] = useState([])
  const [thresholdECDSA, setThresholdECDSA] = useState(null)

  useEffect(() => {
    // Mock network statistics
    setNetworkStats({
      blockHeight: 1234567,
      networkHashRate: '2.3 TH/s',
      activeNodes: 847,
      totalTransactions: 98765432,
      avgBlockTime: 2.1,
      networkUptime: 99.97,
      cyclesBalance: '15.2T',
      totalCanisters: 3
    })

    // Mock canister metrics
    setCanisterMetrics([
      {
        name: 'AI Inference',
        id: 'rdmx6-jaaaa-aaaah-qdhzq-cai',
        status: 'running',
        cycles: '5.2T',
        memory: '2.1GB',
        calls: 15420,
        uptime: 99.98
      },
      {
        name: 'Federated Aggregator',
        id: 'rrkah-fqaaa-aaaah-qdhza-cai',
        status: 'running',
        cycles: '7.8T',
        memory: '1.8GB',
        calls: 8934,
        uptime: 99.95
      },
      {
        name: 'Privacy Engine',
        id: 'renrk-eyaaa-aaaah-qdhzb-cai',
        status: 'running',
        cycles: '2.2T',
        memory: '0.9GB',
        calls: 12567,
        uptime: 99.99
      }
    ])

    // Mock transaction history
    setTransactionHistory([
      { time: '00:00', transactions: 45, cycles: 2.1 },
      { time: '04:00', transactions: 67, cycles: 3.2 },
      { time: '08:00', transactions: 123, cycles: 5.8 },
      { time: '12:00', transactions: 189, cycles: 8.9 },
      { time: '16:00', transactions: 156, cycles: 7.3 },
      { time: '20:00', transactions: 98, cycles: 4.6 },
    ])

    // Mock threshold-ECDSA status
    setThresholdECDSA({
      status: 'active',
      keyShares: 13,
      threshold: 9,
      lastRotation: Date.now() - 86400000, // 24 hours ago
      signaturesGenerated: 2847,
      verificationSuccess: 99.98
    })
  }, [])

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-green-400'
      case 'stopped': return 'text-red-400'
      case 'upgrading': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <CheckCircle className="w-4 h-4" />
      case 'stopped': return <AlertTriangle className="w-4 h-4" />
      case 'upgrading': return <Clock className="w-4 h-4" />
      default: return <Database className="w-4 h-4" />
    }
  }

  if (!networkStats) return <div>Loading blockchain status...</div>

  return (
    <div className="text-white">
      <div className="flex items-center mb-6">
        <Database className="w-8 h-8 mr-3 text-blue-400" />
        <h2 className="text-2xl font-bold">Internet Computer Network Status</h2>
      </div>

      {/* Network Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            <span className="text-xs text-green-400">LIVE</span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {networkStats.blockHeight.toLocaleString()}
          </div>
          <div className="text-sm text-white/60">Block Height</div>
          <div className="text-xs text-white/40 mt-1">
            ~{networkStats.avgBlockTime}s avg time
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Users className="w-6 h-6 text-blue-400" />
            <TrendingUp className="w-4 h-4 text-green-400" />
          </div>
          <div className="text-2xl font-bold mb-1">
            {networkStats.activeNodes}
          </div>
          <div className="text-sm text-white/60">Active Nodes</div>
          <div className="text-xs text-white/40 mt-1">
            {networkStats.networkUptime}% uptime
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Database className="w-6 h-6 text-purple-400" />
            <span className="text-xs text-blue-400">ICP</span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {networkStats.cyclesBalance}
          </div>
          <div className="text-sm text-white/60">Cycles Balance</div>
          <div className="text-xs text-white/40 mt-1">
            {networkStats.totalCanisters} canisters
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <Shield className="w-6 h-6 text-green-400" />
            <span className="text-xs text-green-400">SECURE</span>
          </div>
          <div className="text-2xl font-bold mb-1">
            {(networkStats.totalTransactions / 1000000).toFixed(1)}M
          </div>
          <div className="text-sm text-white/60">Total Transactions</div>
          <div className="text-xs text-white/40 mt-1">
            All cryptographically verified
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Transaction Activity */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Network Activity (24h)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={transactionHistory}>
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
                <Area 
                  type="monotone" 
                  dataKey="transactions" 
                  stroke="#60a5fa" 
                  fill="rgba(96, 165, 250, 0.2)"
                  name="Transactions"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Threshold-ECDSA Status */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-xl font-semibold mb-4">Threshold-ECDSA Security</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-white/60">Status</span>
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                <span className="text-green-400 capitalize">{thresholdECDSA.status}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-white/60">Key Shares</span>
              <span>{thresholdECDSA.keyShares} total</span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-white/60">Threshold</span>
              <span>{thresholdECDSA.threshold} of {thresholdECDSA.keyShares}</span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-white/60">Signatures Generated</span>
              <span>{thresholdECDSA.signaturesGenerated.toLocaleString()}</span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-white/60">Verification Success</span>
              <span className="text-green-400">{thresholdECDSA.verificationSuccess}%</span>
            </div>

            <div className="pt-4 border-t border-white/10">
              <div className="text-sm text-white/60">
                Last Key Rotation: {new Date(thresholdECDSA.lastRotation).toLocaleDateString()}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Canister Status */}
      <div className="bg-white/5 rounded-lg p-6 border border-white/10">
        <h3 className="text-xl font-semibold mb-6">MedChain AI Canisters</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {canisterMetrics.map((canister) => (
            <div key={canister.id} className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium">{canister.name}</h4>
                <div className={`flex items-center ${getStatusColor(canister.status)}`}>
                  {getStatusIcon(canister.status)}
                  <span className="ml-1 text-xs capitalize">{canister.status}</span>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/60">Canister ID:</span>
                  <span className="font-mono text-xs">{canister.id.slice(0, 8)}...</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-white/60">Cycles:</span>
                  <span className="text-blue-400">{canister.cycles}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-white/60">Memory:</span>
                  <span>{canister.memory}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-white/60">Total Calls:</span>
                  <span>{canister.calls.toLocaleString()}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-white/60">Uptime:</span>
                  <span className="text-green-400">{canister.uptime}%</span>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-white/10">
                <div className="w-full bg-white/10 rounded-full h-1">
                  <div 
                    className="bg-gradient-to-r from-green-400 to-blue-500 h-1 rounded-full"
                    style={{ width: `${canister.uptime}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Network Health Indicators */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h4 className="font-medium mb-3 flex items-center">
            <Shield className="w-5 h-5 mr-2 text-green-400" />
            Security Status
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex items-center">
              <CheckCircle className="w-4 h-4 text-green-400 mr-2" />
              <span>Threshold-ECDSA Active</span>
            </div>
            <div className="flex items-center">
              <CheckCircle className="w-4 h-4 text-green-400 mr-2" />
              <span>All Signatures Verified</span>
            </div>
            <div className="flex items-center">
              <CheckCircle className="w-4 h-4 text-green-400 mr-2" />
              <span>Network Consensus Healthy</span>
            </div>
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h4 className="font-medium mb-3 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-yellow-400" />
            Performance
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-white/60">Avg Response Time:</span>
              <span className="text-green-400">1.2s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/60">Throughput:</span>
              <span className="text-blue-400">450 TPS</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/60">Success Rate:</span>
              <span className="text-green-400">99.97%</span>
            </div>
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h4 className="font-medium mb-3 flex items-center">
            <Database className="w-5 h-5 mr-2 text-purple-400" />
            Resources
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-white/60">Total Cycles:</span>
              <span className="text-purple-400">{networkStats.cyclesBalance}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/60">Memory Usage:</span>
              <span className="text-blue-400">4.8GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/60">Storage:</span>
              <span className="text-green-400">2.1TB</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BlockchainStatus