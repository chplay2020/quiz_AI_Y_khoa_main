import { useState, useEffect } from 'react'
import { Question, QuestionOption } from '../api'
import { XMarkIcon, PlusIcon, TrashIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface QuestionEditModalProps {
    question: Question
    isOpen: boolean
    onClose: () => void
    onSave: (questionId: string, updates: Partial<Question>) => void
}

export default function QuestionEditModal({
    question,
    isOpen,
    onClose,
    onSave,
}: QuestionEditModalProps) {
    const [editedQuestion, setEditedQuestion] = useState<Partial<Question>>({})
    const [options, setOptions] = useState<QuestionOption[]>([])

    useEffect(() => {
        if (question) {
            setEditedQuestion({
                question_text: question.question_text,
                explanation: question.explanation,
                difficulty: question.difficulty,
                topic: question.topic,
                correct_answer: question.correct_answer,
            })
            setOptions(question.options?.map(opt => ({ ...opt })) || [])
        }
    }, [question])

    if (!isOpen) return null

    const handleOptionChange = (index: number, field: keyof QuestionOption, value: string | boolean) => {
        const newOptions = [...options]
        newOptions[index] = { ...newOptions[index], [field]: value }

        // Nếu đánh dấu is_correct, cập nhật correct_answer và bỏ is_correct của các option khác
        if (field === 'is_correct' && value === true) {
            newOptions.forEach((opt, i) => {
                opt.is_correct = i === index
            })
            setEditedQuestion(prev => ({ ...prev, correct_answer: newOptions[index].id }))
        }

        setOptions(newOptions)
    }

    const addOption = () => {
        const nextId = String.fromCharCode(65 + options.length) // A, B, C, D, E...
        setOptions([...options, { id: nextId, text: '', is_correct: false }])
    }

    const removeOption = (index: number) => {
        if (options.length <= 2) return // Tối thiểu 2 options
        const newOptions = options.filter((_, i) => i !== index)
        // Re-label options
        newOptions.forEach((opt, i) => {
            opt.id = String.fromCharCode(65 + i)
        })
        setOptions(newOptions)
    }

    const handleSave = () => {
        const updates: Partial<Question> = {
            ...editedQuestion,
            options: options,
        }
        onSave(question.id, updates)
        onClose()
    }

    return (
        <div className="fixed inset-0 z-50 overflow-y-auto">
            <div className="flex min-h-screen items-center justify-center p-4">
                {/* Backdrop */}
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
                    onClick={onClose}
                />

                {/* Modal */}
                <div className="relative bg-white rounded-xl shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b sticky top-0 bg-white z-10">
                        <h2 className="text-lg font-semibold text-gray-900">
                            ✏️ Sửa câu hỏi
                        </h2>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                            <XMarkIcon className="w-5 h-5 text-gray-500" />
                        </button>
                    </div>

                    {/* AI Review suggestions */}
                    {question.ai_review && (question.ai_review.status === 'needs_revision' || question.ai_review.status === 'reject') && (
                        <div className={clsx(
                            'mx-4 mt-4 p-3 rounded-lg border',
                            question.ai_review.status === 'needs_revision' ? 'bg-yellow-50 border-yellow-200' : 'bg-red-50 border-red-200'
                        )}>
                            <p className="text-sm font-medium text-gray-700 mb-2">
                                💡 Gợi ý từ AI Double-Check:
                            </p>
                            {question.ai_review.issues && question.ai_review.issues.length > 0 && (
                                <div className="mb-2">
                                    <p className="text-xs font-medium text-red-700">Vấn đề:</p>
                                    <ul className="text-xs text-red-600 list-disc list-inside">
                                        {question.ai_review.issues.map((issue, i) => (
                                            <li key={i}>{issue}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {question.ai_review.suggestions && question.ai_review.suggestions.length > 0 && (
                                <div>
                                    <p className="text-xs font-medium text-blue-700">Gợi ý:</p>
                                    <ul className="text-xs text-blue-600 list-disc list-inside">
                                        {question.ai_review.suggestions.map((suggestion, i) => (
                                            <li key={i}>{suggestion}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {question.ai_review.corrected_answer && (
                                <div className="mt-2 p-2 bg-green-100 rounded">
                                    <p className="text-xs font-medium text-green-700">
                                        Đáp án đề xuất: {question.ai_review.corrected_answer}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Content */}
                    <div className="p-4 space-y-4">
                        {/* Question text */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Nội dung câu hỏi *
                            </label>
                            <textarea
                                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                                rows={4}
                                value={editedQuestion.question_text || ''}
                                onChange={(e) => setEditedQuestion(prev => ({ ...prev, question_text: e.target.value }))}
                                placeholder="Nhập nội dung câu hỏi..."
                            />
                        </div>

                        {/* Options */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Các đáp án * (Chọn đáp án đúng)
                            </label>
                            <div className="space-y-2">
                                {options.map((option, index) => (
                                    <div key={index} className="flex items-center gap-2">
                                        <input
                                            type="radio"
                                            name="correct_answer"
                                            checked={option.is_correct}
                                            onChange={() => handleOptionChange(index, 'is_correct', true)}
                                            className="w-4 h-4 text-primary-600"
                                        />
                                        <span className="w-8 h-8 flex items-center justify-center bg-gray-200 rounded-full font-medium text-sm">
                                            {option.id}
                                        </span>
                                        <input
                                            type="text"
                                            className={clsx(
                                                'flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent',
                                                option.is_correct ? 'border-green-500 bg-green-50' : 'border-gray-300'
                                            )}
                                            value={option.text}
                                            onChange={(e) => handleOptionChange(index, 'text', e.target.value)}
                                            placeholder={`Nội dung đáp án ${option.id}...`}
                                        />
                                        {options.length > 2 && (
                                            <button
                                                onClick={() => removeOption(index)}
                                                className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                                            >
                                                <TrashIcon className="w-4 h-4" />
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                            {options.length < 6 && (
                                <button
                                    onClick={addOption}
                                    className="mt-2 flex items-center text-sm text-primary-600 hover:text-primary-700"
                                >
                                    <PlusIcon className="w-4 h-4 mr-1" />
                                    Thêm đáp án
                                </button>
                            )}
                        </div>

                        {/* Explanation */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Giải thích đáp án
                            </label>
                            <textarea
                                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                                rows={3}
                                value={editedQuestion.explanation || ''}
                                onChange={(e) => setEditedQuestion(prev => ({ ...prev, explanation: e.target.value }))}
                                placeholder="Giải thích tại sao đáp án đúng..."
                            />
                        </div>

                        {/* Difficulty & Topic */}
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Độ khó
                                </label>
                                <select
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                    value={editedQuestion.difficulty || 'medium'}
                                    onChange={(e) => setEditedQuestion(prev => ({ ...prev, difficulty: e.target.value as 'easy' | 'medium' | 'hard' }))}
                                >
                                    <option value="easy">Dễ</option>
                                    <option value="medium">Trung bình</option>
                                    <option value="hard">Khó</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Chủ đề
                                </label>
                                <input
                                    type="text"
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                    value={editedQuestion.topic || ''}
                                    onChange={(e) => setEditedQuestion(prev => ({ ...prev, topic: e.target.value }))}
                                    placeholder="Chủ đề câu hỏi..."
                                />
                            </div>
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="flex items-center justify-end gap-3 p-4 border-t sticky bottom-0 bg-white">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                        >
                            Hủy
                        </button>
                        <button
                            onClick={handleSave}
                            className="px-4 py-2 text-white bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors"
                        >
                            💾 Lưu thay đổi
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
