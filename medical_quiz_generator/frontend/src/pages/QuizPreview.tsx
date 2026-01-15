import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { Question } from '../api'
import QuestionCard from '../components/QuestionCard'

// Mock quiz data - in production, fetch from API
const mockQuestions: Question[] = []

export default function QuizPreview() {
    const { quizId } = useParams()
    const [currentIndex, setCurrentIndex] = useState(0)
    const [answers, setAnswers] = useState<Record<string, string>>({})
    const [showResults, setShowResults] = useState(false)

    const questions = mockQuestions // Replace with actual query

    if (questions.length === 0) {
        return (
            <div className="card text-center py-12">
                <p className="text-gray-500">Không tìm thấy bài quiz</p>
            </div>
        )
    }

    const currentQuestion = questions[currentIndex]
    const isLastQuestion = currentIndex === questions.length - 1

    const handleAnswer = (answerId: string) => {
        setAnswers(prev => ({
            ...prev,
            [currentQuestion.id]: answerId
        }))
    }

    const handleNext = () => {
        if (isLastQuestion) {
            setShowResults(true)
        } else {
            setCurrentIndex(prev => prev + 1)
        }
    }

    const calculateScore = () => {
        let correct = 0
        questions.forEach(q => {
            if (answers[q.id] === q.correct_answer) {
                correct++
            }
        })
        return { correct, total: questions.length }
    }

    if (showResults) {
        const { correct, total } = calculateScore()
        const percentage = Math.round((correct / total) * 100)

        return (
            <div className="max-w-2xl mx-auto space-y-6">
                <div className="card text-center">
                    <h1 className="text-2xl font-bold text-gray-900 mb-4">Kết quả</h1>
                    <div className="text-6xl font-bold text-primary-600 mb-4">{percentage}%</div>
                    <p className="text-gray-600 mb-6">
                        Trả lời đúng {correct}/{total} câu hỏi
                    </p>
                    <button
                        onClick={() => {
                            setCurrentIndex(0)
                            setAnswers({})
                            setShowResults(false)
                        }}
                        className="btn-primary"
                    >
                        Làm lại
                    </button>
                </div>

                {/* Review questions */}
                <div className="space-y-4">
                    <h2 className="text-xl font-bold text-gray-900">Xem lại đáp án</h2>
                    {questions.map((q, index) => (
                        <QuestionCard
                            key={q.id}
                            question={q}
                            questionNumber={index + 1}
                            mode="review"
                            selectedAnswer={answers[q.id]}
                        />
                    ))}
                </div>
            </div>
        )
    }

    return (
        <div className="max-w-2xl mx-auto space-y-6">
            {/* Progress */}
            <div className="card">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-600">
                        Câu {currentIndex + 1} / {questions.length}
                    </span>
                    <span className="text-sm text-gray-500">
                        {Object.keys(answers).length} đã trả lời
                    </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                        className="bg-primary-600 h-2 rounded-full transition-all"
                        style={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }}
                    />
                </div>
            </div>

            {/* Question */}
            <QuestionCard
                question={currentQuestion}
                questionNumber={currentIndex + 1}
                mode="quiz"
                selectedAnswer={answers[currentQuestion.id]}
                onAnswerSelect={handleAnswer}
            />

            {/* Navigation */}
            <div className="flex justify-between">
                <button
                    onClick={() => setCurrentIndex(prev => prev - 1)}
                    disabled={currentIndex === 0}
                    className="btn-secondary disabled:opacity-50"
                >
                    Câu trước
                </button>
                <button
                    onClick={handleNext}
                    disabled={!answers[currentQuestion.id]}
                    className="btn-primary disabled:opacity-50"
                >
                    {isLastQuestion ? 'Hoàn thành' : 'Câu tiếp'}
                </button>
            </div>
        </div>
    )
}
