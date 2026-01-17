import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    MagnifyingGlassIcon,
    FunnelIcon,
    TrashIcon,
    ArrowDownTrayIcon,
    PencilIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import { questionsApi, Question } from '../api'
import { useAppStore } from '../store'
import QuestionCard from '../components/QuestionCard'
import QuestionEditModal from '../components/QuestionEditModal'

export default function Questions() {
    const [searchQuery, setSearchQuery] = useState('')
    const [filters, setFilters] = useState({
        difficulty: '',
        question_type: '',
        topic: '',
    })
    const [showFilters, setShowFilters] = useState(false)
    const [editingQuestion, setEditingQuestion] = useState<Question | null>(null)
    const [isEditModalOpen, setIsEditModalOpen] = useState(false)

    const queryClient = useQueryClient()
    const { selectedQuestions, toggleQuestionSelection, clearQuestionSelection } = useAppStore()

    const { data, isLoading } = useQuery({
        queryKey: ['questions', filters],
        queryFn: () => questionsApi.list(filters),
    })

    const deleteMutation = useMutation({
        mutationFn: questionsApi.delete,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['questions'] })
            toast.success('Đã xóa câu hỏi')
        },
    })

    const updateMutation = useMutation({
        mutationFn: ({ id, updates }: { id: string; updates: Partial<Question> }) =>
            questionsApi.update(id, updates),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['questions'] })
            toast.success('Đã cập nhật câu hỏi')
            setEditingQuestion(null)
            setIsEditModalOpen(false)
        },
    })

    const handleEditQuestion = (question: Question) => {
        setEditingQuestion(question)
        setIsEditModalOpen(true)
    }

    const handleSaveQuestion = (questionId: string, updates: Partial<Question>) => {
        updateMutation.mutate({ id: questionId, updates })
    }

    const questions: Question[] = data?.data?.questions || []

    const filteredQuestions = searchQuery
        ? questions.filter(q =>
            q.question_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
            q.topic?.toLowerCase().includes(searchQuery.toLowerCase())
        )
        : questions

    const handleExport = async (format: 'json' | 'word' | 'pdf' | 'excel') => {
        if (selectedQuestions.length === 0) {
            toast.error('Vui lòng chọn ít nhất một câu hỏi để xuất')
            return
        }

        try {
            const result = await questionsApi.export(selectedQuestions, format, {
                include_answers: true,
                include_explanations: true,
            })

            if (format === 'json') {
                // Download JSON
                const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `medical_quiz_${Date.now()}.json`
                a.click()
                URL.revokeObjectURL(url)
                toast.success('Đã xuất file JSON')
            } else if (format === 'word') {
                // Download Word file from base64
                const fileContent = result.data.file_content
                const filename = result.data.filename || `cau_hoi_trac_nghiem_${Date.now()}.docx`

                // Decode base64 to blob
                const byteCharacters = atob(fileContent)
                const byteNumbers = new Array(byteCharacters.length)
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i)
                }
                const byteArray = new Uint8Array(byteNumbers)
                const blob = new Blob([byteArray], {
                    type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                })

                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = filename
                a.click()
                URL.revokeObjectURL(url)
                toast.success('Đã xuất file Word')
            }
        } catch (error) {
            toast.error('Không thể xuất file')
        }
    }

    const handleBulkDelete = async () => {
        if (selectedQuestions.length === 0) {
            toast.error('Vui lòng chọn ít nhất một câu hỏi')
            return
        }

        if (!confirm(`Bạn có chắc muốn xóa ${selectedQuestions.length} câu hỏi?`)) {
            return
        }

        for (const id of selectedQuestions) {
            await deleteMutation.mutateAsync(id)
        }

        clearQuestionSelection()
        toast.success(`Đã xóa ${selectedQuestions.length} câu hỏi`)
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Ngân hàng câu hỏi</h1>
                    <p className="text-gray-600 mt-1">
                        {questions.length} câu hỏi • {selectedQuestions.length} đã chọn
                    </p>
                </div>
                <div className="flex flex-wrap gap-2">
                    <button
                        onClick={() => handleExport('json')}
                        className="btn-secondary flex items-center"
                        disabled={selectedQuestions.length === 0}
                    >
                        <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                        Xuất JSON
                    </button>
                    <button
                        onClick={() => handleExport('word')}
                        className="btn-secondary flex items-center"
                        disabled={selectedQuestions.length === 0}
                    >
                        <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                        Xuất Word
                    </button>
                    <button
                        onClick={handleBulkDelete}
                        className="btn-secondary flex items-center text-red-600 hover:bg-red-50"
                        disabled={selectedQuestions.length === 0}
                    >
                        <TrashIcon className="w-4 h-4 mr-2" />
                        Xóa ({selectedQuestions.length})
                    </button>
                </div>
            </div>

            {/* Search and Filters */}
            <div className="card">
                <div className="flex flex-col sm:flex-row gap-4">
                    {/* Search */}
                    <div className="flex-1 relative">
                        <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            className="input pl-10"
                            placeholder="Tìm kiếm câu hỏi..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>

                    {/* Filter Toggle */}
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={clsx(
                            'btn-secondary flex items-center',
                            showFilters && 'bg-primary-50 border-primary-300'
                        )}
                    >
                        <FunnelIcon className="w-4 h-4 mr-2" />
                        Bộ lọc
                    </button>
                </div>

                {/* Filters */}
                {showFilters && (
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 pt-4 border-t">
                        <div>
                            <label className="label">Độ khó</label>
                            <select
                                className="input"
                                value={filters.difficulty}
                                onChange={(e) => setFilters(prev => ({ ...prev, difficulty: e.target.value }))}
                            >
                                <option value="">Tất cả</option>
                                <option value="easy">Dễ</option>
                                <option value="medium">Trung bình</option>
                                <option value="hard">Khó</option>
                            </select>
                        </div>
                        <div>
                            <label className="label">Loại câu hỏi</label>
                            <select
                                className="input"
                                value={filters.question_type}
                                onChange={(e) => setFilters(prev => ({ ...prev, question_type: e.target.value }))}
                            >
                                <option value="">Tất cả</option>
                                <option value="single_choice">Chọn một đáp án</option>
                                <option value="multiple_choice">Chọn nhiều đáp án</option>
                                <option value="case_based">Tình huống lâm sàng</option>
                            </select>
                        </div>
                        <div>
                            <label className="label">Chủ đề</label>
                            <input
                                type="text"
                                className="input"
                                placeholder="Nhập chủ đề..."
                                value={filters.topic}
                                onChange={(e) => setFilters(prev => ({ ...prev, topic: e.target.value }))}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Questions List */}
            {isLoading ? (
                <div className="text-center py-12">
                    <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
                    <p className="text-gray-500 mt-4">Đang tải...</p>
                </div>
            ) : filteredQuestions.length === 0 ? (
                <div className="card text-center py-12">
                    <p className="text-gray-500">Không tìm thấy câu hỏi nào</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {/* Select All */}
                    <div className="flex items-center gap-4">
                        <label className="flex items-center">
                            <input
                                type="checkbox"
                                className="rounded border-gray-300 text-primary-600 mr-2"
                                checked={selectedQuestions.length === filteredQuestions.length && filteredQuestions.length > 0}
                                onChange={(e) => {
                                    if (e.target.checked) {
                                        filteredQuestions.forEach(q => {
                                            if (!selectedQuestions.includes(q.id)) {
                                                toggleQuestionSelection(q.id)
                                            }
                                        })
                                    } else {
                                        clearQuestionSelection()
                                    }
                                }}
                            />
                            <span className="text-sm text-gray-600">Chọn tất cả</span>
                        </label>
                    </div>

                    {/* Questions */}
                    {filteredQuestions.map((question, index) => (
                        <div key={`q-${question.id}-${index}`} className="relative">
                            {/* Selection checkbox */}
                            <div className="absolute top-6 left-6 z-10">
                                <input
                                    type="checkbox"
                                    className="rounded border-gray-300 text-primary-600"
                                    checked={selectedQuestions.includes(question.id)}
                                    onChange={() => toggleQuestionSelection(question.id)}
                                />
                            </div>

                            {/* Actions */}
                            <div className="absolute top-6 right-6 z-10 flex gap-2">
                                <button
                                    onClick={() => handleEditQuestion(question)}
                                    className="p-2 hover:bg-gray-100 rounded-lg"
                                    title="Chỉnh sửa"
                                >
                                    <PencilIcon className="w-4 h-4 text-gray-500" />
                                </button>
                                <button
                                    onClick={() => {
                                        if (confirm('Bạn có chắc muốn xóa câu hỏi này?')) {
                                            deleteMutation.mutate(question.id)
                                        }
                                    }}
                                    className="p-2 hover:bg-red-100 rounded-lg"
                                    title="Xóa"
                                >
                                    <TrashIcon className="w-4 h-4 text-red-500" />
                                </button>
                            </div>

                            <div className={clsx(
                                'pl-12',
                                selectedQuestions.includes(question.id) && 'ring-2 ring-primary-500 rounded-xl'
                            )}>
                                <QuestionCard
                                    question={question}
                                    questionNumber={index + 1}
                                    mode="preview"
                                    showAnswer={true}
                                    onEdit={handleEditQuestion}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Edit Modal */}
            {editingQuestion && (
                <QuestionEditModal
                    question={editingQuestion}
                    isOpen={isEditModalOpen}
                    onClose={() => {
                        setIsEditModalOpen(false)
                        setEditingQuestion(null)
                    }}
                    onSave={handleSaveQuestion}
                />
            )}
        </div>
    )
}
