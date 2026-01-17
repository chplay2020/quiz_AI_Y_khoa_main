import { useState } from 'react'
import { Question } from '../api'
import {
    CheckCircleIcon,
    XCircleIcon,
    LightBulbIcon,
    ShieldCheckIcon,
    ExclamationTriangleIcon,
    PencilIcon
} from '@heroicons/react/24/solid'
import clsx from 'clsx'

interface QuestionCardProps {
    question: Question
    showAnswer?: boolean
    mode?: 'preview' | 'quiz' | 'review'
    questionNumber?: number
    onAnswerSelect?: (answerId: string) => void
    selectedAnswer?: string
    showAIReview?: boolean
    onEdit?: (question: Question) => void
}

export default function QuestionCard({
    question,
    showAnswer = false,
    mode = 'preview',
    questionNumber,
    onAnswerSelect,
    selectedAnswer,
    showAIReview = true,
    onEdit,
}: QuestionCardProps) {
    const [localShowAnswer, setLocalShowAnswer] = useState(showAnswer)
    const [localSelectedAnswer, setLocalSelectedAnswer] = useState<string | null>(selectedAnswer || null)

    // Safety checks - return null if question data is invalid
    if (!question || !question.question_text) {
        return (
            <div className="card bg-yellow-50 border-yellow-200">
                <p className="text-yellow-700">Câu hỏi không hợp lệ hoặc đang tải...</p>
            </div>
        )
    }

    // Ensure options is always an array
    const options = question.options || []
    const difficulty = question.difficulty || 'medium'

    const handleOptionClick = (optionId: string) => {
        if (mode === 'quiz') {
            setLocalSelectedAnswer(optionId)
            onAnswerSelect?.(optionId)
        }
    }

    const getDifficultyColor = (difficulty: string) => {
        switch (difficulty) {
            case 'easy':
                return 'bg-green-100 text-green-800'
            case 'medium':
                return 'bg-yellow-100 text-yellow-800'
            case 'hard':
                return 'bg-red-100 text-red-800'
            default:
                return 'bg-gray-100 text-gray-800'
        }
    }

    const getDifficultyLabel = (difficulty: string) => {
        switch (difficulty) {
            case 'easy':
                return 'Dễ'
            case 'medium':
                return 'Trung bình'
            case 'hard':
                return 'Khó'
            default:
                return difficulty
        }
    }

    return (
        <div className="card">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                {/* Left: Question number, difficulty, topic */}
                <div className="flex items-center space-x-3">
                    {questionNumber && (
                        <span className="flex items-center justify-center w-8 h-8 bg-primary-100 text-primary-700 rounded-full font-semibold text-sm">
                            {questionNumber}
                        </span>
                    )}
                    <span className={clsx('px-2 py-1 rounded-full text-xs font-medium', getDifficultyColor(difficulty))}>
                        {getDifficultyLabel(difficulty)}
                    </span>
                    {question.topic && (
                        <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">
                            {question.topic}
                        </span>
                    )}
                </div>

                {/* Center: Case-based badge */}
                {question.question_type === 'case_based' && (
                    <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                        Tình huống lâm sàng
                    </span>
                )}

                {/* Right: Empty placeholder for balance */}
                <div className="w-8"></div>
            </div>

            {/* Question text */}
            <p className="text-gray-900 font-medium mb-4 leading-relaxed">
                {question.question_text}
            </p>

            {/* Options */}
            <div className="space-y-2">
                {options.length > 0 ? options.map((option) => {
                    const isSelected = localSelectedAnswer === option.id
                    const isCorrect = option.is_correct
                    const showResult = localShowAnswer || (mode === 'review')

                    return (
                        <button
                            key={option.id || Math.random().toString()}
                            onClick={() => handleOptionClick(option.id)}
                            disabled={mode === 'preview' || localShowAnswer}
                            className={clsx(
                                'w-full flex items-center p-4 rounded-lg border-2 text-left transition-all',
                                !showResult && !isSelected && 'border-gray-200 hover:border-primary-300 hover:bg-primary-50',
                                !showResult && isSelected && 'border-primary-500 bg-primary-50',
                                showResult && isCorrect && 'border-green-500 bg-green-50',
                                showResult && !isCorrect && isSelected && 'border-red-500 bg-red-50',
                                showResult && !isCorrect && !isSelected && 'border-gray-200 opacity-60',
                            )}
                        >
                            <span className={clsx(
                                'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-medium mr-3',
                                !showResult && 'bg-gray-200 text-gray-700',
                                showResult && isCorrect && 'bg-green-500 text-white',
                                showResult && !isCorrect && isSelected && 'bg-red-500 text-white',
                                showResult && !isCorrect && !isSelected && 'bg-gray-200 text-gray-500',
                            )}>
                                {option.id}
                            </span>
                            <span className="flex-1">{option.text}</span>
                            {showResult && (
                                <span className="ml-2">
                                    {isCorrect ? (
                                        <CheckCircleIcon className="w-6 h-6 text-green-500" />
                                    ) : isSelected ? (
                                        <XCircleIcon className="w-6 h-6 text-red-500" />
                                    ) : null}
                                </span>
                            )}
                        </button>
                    )
                }) : (
                    <p className="text-gray-500 text-sm">Không có lựa chọn nào</p>
                )}
            </div>

            {/* Show answer button (for quiz mode) */}
            {mode === 'quiz' && localSelectedAnswer && !localShowAnswer && (
                <button
                    onClick={() => setLocalShowAnswer(true)}
                    className="mt-4 btn-primary w-full"
                >
                    Xem đáp án
                </button>
            )}

            {/* Explanation */}
            {(localShowAnswer || mode === 'review') && question.explanation && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-start">
                        <LightBulbIcon className="w-5 h-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="text-sm font-medium text-blue-800 mb-1">Giải thích:</p>
                            <p className="text-sm text-blue-700">{question.explanation}</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Reference text */}
            {mode === 'preview' && question.reference_text && (
                <details className="mt-4">
                    <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
                        Xem nguồn tham khảo
                    </summary>
                    <p className="mt-2 text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
                        {question.reference_text}
                    </p>
                </details>
            )}

            {/* AI Review Section */}
            {showAIReview && question.ai_review && question.ai_review.reviewed && (
                <div className={clsx(
                    'mt-4 p-4 rounded-lg border',
                    question.ai_review.status === 'approved' && 'bg-green-50 border-green-200',
                    question.ai_review.status === 'needs_revision' && 'bg-yellow-50 border-yellow-200',
                    question.ai_review.status === 'reject' && 'bg-red-50 border-red-200',
                    question.ai_review.status === 'error' && 'bg-gray-50 border-gray-200',
                )}>
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center">
                            {question.ai_review.status === 'approved' ? (
                                <ShieldCheckIcon className="w-5 h-5 text-green-600 mr-2" />
                            ) : (
                                <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 mr-2" />
                            )}
                            <span className={clsx(
                                'text-sm font-medium',
                                question.ai_review.status === 'approved' && 'text-green-800',
                                question.ai_review.status === 'needs_revision' && 'text-yellow-800',
                                question.ai_review.status === 'reject' && 'text-red-800',
                            )}>
                                AI Double-Check: {
                                    question.ai_review.status === 'approved' ? 'Đạt chuẩn' :
                                        question.ai_review.status === 'needs_revision' ? 'Cần sửa' :
                                            question.ai_review.status === 'reject' ? 'Không đạt' : 'Lỗi'
                                }
                            </span>
                        </div>

                        {/* Scores */}
                        <div className="flex items-center space-x-3 text-xs">
                            {question.ai_review.accuracy_score && (
                                <span className={clsx(
                                    'px-2 py-1 rounded-full',
                                    question.ai_review.accuracy_score >= 8 ? 'bg-green-200 text-green-800' :
                                        question.ai_review.accuracy_score >= 6 ? 'bg-yellow-200 text-yellow-800' :
                                            'bg-red-200 text-red-800'
                                )}>
                                    Chính xác: {question.ai_review.accuracy_score}/10
                                </span>
                            )}
                            {question.ai_review.clarity_score && (
                                <span className="px-2 py-1 rounded-full bg-blue-100 text-blue-800">
                                    Rõ ràng: {question.ai_review.clarity_score}/10
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Issues */}
                    {question.ai_review.issues && question.ai_review.issues.length > 0 && (
                        <div className="mb-2">
                            <p className="text-xs font-medium text-gray-700 mb-1">Vấn đề:</p>
                            <ul className="text-xs text-gray-600 list-disc list-inside">
                                {question.ai_review.issues.map((issue, i) => (
                                    <li key={i}>{issue}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Suggestions */}
                    {question.ai_review.suggestions && question.ai_review.suggestions.length > 0 && (
                        <div>
                            <p className="text-xs font-medium text-gray-700 mb-1">Gợi ý cải thiện:</p>
                            <ul className="text-xs text-gray-600 list-disc list-inside">
                                {question.ai_review.suggestions.map((suggestion, i) => (
                                    <li key={i}>{suggestion}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Edit button for needs_revision or reject */}
                    {(question.ai_review.status === 'needs_revision' || question.ai_review.status === 'reject') && onEdit && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                            <button
                                onClick={() => onEdit(question)}
                                className={clsx(
                                    'w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
                                    question.ai_review.status === 'needs_revision'
                                        ? 'bg-yellow-500 hover:bg-yellow-600 text-white'
                                        : 'bg-red-500 hover:bg-red-600 text-white'
                                )}
                            >
                                <PencilIcon className="w-4 h-4" />
                                Sửa câu hỏi này
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
