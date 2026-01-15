import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Questions from './pages/Questions'
import Generate from './pages/Generate'
import QuizPreview from './pages/QuizPreview'

function App() {
    return (
        <Layout>
            <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/documents" element={<Documents />} />
                <Route path="/generate" element={<Generate />} />
                <Route path="/questions" element={<Questions />} />
                <Route path="/quiz/:quizId" element={<QuizPreview />} />
            </Routes>
        </Layout>
    )
}

export default App
