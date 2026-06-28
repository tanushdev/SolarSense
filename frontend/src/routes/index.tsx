import { createBrowserRouter, Navigate } from 'react-router-dom'
import { AppLayout } from '@/layouts/AppLayout'
import DashboardPage from '@/pages/Dashboard/DashboardPage'
import LiveMonitoringPage from '@/pages/LiveMonitoring/LiveMonitoringPage'
import ForecastsPage from '@/pages/Forecasts/ForecastsPage'
import HistoricalEventsPage from '@/pages/HistoricalEvents/HistoricalEventsPage'
import ModelPerformancePage from '@/pages/ModelPerformance/ModelPerformancePage'
import DatasetsPage from '@/pages/Datasets/DatasetsPage'
import ValidationPage from '@/pages/Validation/ValidationPage'

export const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { path: '/', element: <DashboardPage /> },
      { path: '/live-monitor', element: <LiveMonitoringPage /> },
      { path: '/forecasts', element: <ForecastsPage /> },
      { path: '/historical-events', element: <HistoricalEventsPage /> },
      { path: '/model-performance', element: <ModelPerformancePage /> },
      { path: '/datasets', element: <DatasetsPage /> },
      { path: '/validation', element: <ValidationPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])
