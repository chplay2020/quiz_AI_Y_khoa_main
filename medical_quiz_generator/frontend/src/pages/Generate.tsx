import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
    SparklesIcon,
    DocumentTextIcon,
    AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import { documentsApi, questionsApi, GenerationRequest, GenerationStatus, Question } from '../api'
import { useAppStore } from '../store'
import QuestionCard from '../components/QuestionCard'
import QuestionEditModal from '../components/QuestionEditModal'

export default function Generate() {
    const navigate = useNavigate()
    const queryClient = useQueryClient()
    const { selectedDocuments, currentTaskId, setCurrentTaskId } = useAppStore()

    const [config, setConfig] = useState<GenerationRequest>({
        document_ids: [],
        num_questions: 10,
        difficulty: 'medium',
        include_case_based: false,
        include_explanations: true,

    })

    const [generationStatus, setGenerationStatus] = useState<GenerationStatus | null>(null)
    const [isPolling, setIsPolling] = useState(false)

    // Sử dụng useRef để tránh tạo nhiều intervals
    const hasShownToastRef = useRef(false)

    // Edit modal state
    const [editingQuestion, setEditingQuestion] = useState<Question | null>(null)
    const [isEditModalOpen, setIsEditModalOpen] = useState(false)

    // Update question mutation
    const updateMutation = useMutation({
        mutationFn: ({ id, updates }: { id: string; updates: Partial<Question> }) =>
            questionsApi.update(id, updates),
        onSuccess: (data, variables) => {
            // Cập nhật question trong generationStatus
            if (generationStatus?.questions) {
                const updatedQuestions = generationStatus.questions.map(q =>
                    q.id === variables.id ? { ...q, ...variables.updates, ai_review: { ...q.ai_review, status: 'approved' as const, reviewed: true } } : q
                )
                setGenerationStatus(prev => prev ? { ...prev, questions: updatedQuestions } : null)
            }
            queryClient.invalidateQueries({ queryKey: ['questions'] })
            toast.success('Đã cập nhật câu hỏi!')
            setIsEditModalOpen(false)
            setEditingQuestion(null)
        },
        onError: (error: Error) => {
            toast.error(`Lỗi: ${error.message}`)
        },
    })

    // Load documents
    const { data: docsData } = useQuery({
        queryKey: ['documents'],
        queryFn: () => documentsApi.list({ status: 'completed' }),
    })

    const documents = docsData?.data?.documents || []

    // Generation mutation
    const generateMutation = useMutation({
        mutationFn: questionsApi.generate,
        onSuccess: (data) => {
            setCurrentTaskId(data.task_id)
            setIsPolling(true)
            hasShownToastRef.current = false  // Reset ref khi bắt đầu tạo mới
            toast.success('Đang tạo câu hỏi...')
        },
        onError: (error: Error) => {
            toast.error(`Lỗi: ${error.message}`)
        },
    })

    // Poll for status
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>

        if (isPolling && currentTaskId) {
            interval = setInterval(async () => {
                try {
                    const status = await questionsApi.getGenerationStatus(currentTaskId)
                    setGenerationStatus(prev => {
                        if (prev?.questions?.length && !status.questions) {
                            return { ...status, questions: prev.questions }
                        }
                        return status
                    })


                    if (status.status === 'completed' || status.status === 'failed') {
                        clearInterval(interval)
                        setIsPolling(false)

                        // Chỉ hiển thị toast 1 lần duy nhất sử dụng ref
                        if (!hasShownToastRef.current) {
                            hasShownToastRef.current = true

                            if (status.status === 'completed') {
                                const reviewMsg = status.review_stats
                                    ? ` (${status.review_stats.high_accuracy} câu đạt chuẩn)`
                                    : ''
                                toast.success(`Đã tạo ${status.generated_questions} câu hỏi!${reviewMsg}`)
                            } else {
                                toast.error(`Lỗi: ${status.error}`)
                            }
                        }
                    }
                } catch (error) {
                    console.error('Failed to fetch status:', error)
                }
            }, 2000)
        }

        return () => {
            if (interval) clearInterval(interval)
        }
    }, [isPolling, currentTaskId])

    // Update config when selected documents change
    useEffect(() => {
        setConfig(prev => ({ ...prev, document_ids: selectedDocuments }))
    }, [selectedDocuments])

    const handleSubmit = () => {
        if (config.document_ids.length === 0) {
            toast.error('Vui lòng chọn ít nhất một tài liệu')
            return
        }

        setGenerationStatus(null)
        hasShownToastRef.current = false  // Reset ref khi submit lại
        generateMutation.mutate(config)
    }

    const handleEditQuestion = (question: Question) => {
        setEditingQuestion(question)
        setIsEditModalOpen(true)
    }

    const handleSaveQuestion = (questionId: string, updates: Partial<Question>) => {
        updateMutation.mutate({ id: questionId, updates })
    }

    const isGenerating = generateMutation.isPending || isPolling

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-bold text-gray-900">Tạo câu hỏi trắc nghiệm</h1>
                <p className="text-gray-600 mt-1">Chọn tài liệu và cấu hình để tạo câu hỏi bằng AI</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: Configuration */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Document Selection */}
                    <div className="card">
                        <div className="flex items-center mb-4">
                            <DocumentTextIcon className="w-5 h-5 text-primary-600 mr-2" />
                            <h2 className="text-lg font-semibold text-gray-900">Chọn tài liệu</h2>
                        </div>

                        {documents.length === 0 ? (
                            <div className="text-center py-8">
                                <p className="text-gray-500 mb-4">Chưa có tài liệu nào đã xử lý</p>
                                <button
                                    onClick={() => navigate('/documents')}
                                    className="btn-primary"
                                >
                                    Tải lên tài liệu
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {documents.map((doc: any) => (
                                    <label
                                        key={doc.id}
                                        className={clsx(
                                            'flex items-center p-3 rounded-lg border cursor-pointer transition-colors',
                                            config.document_ids.includes(doc.id)
                                                ? 'border-primary-500 bg-primary-50'
                                                : 'border-gray-200 hover:bg-gray-50'
                                        )}
                                    >
                                        <input
                                            type="checkbox"
                                            className="rounded border-gray-300 text-primary-600 mr-3"
                                            checked={config.document_ids.includes(doc.id)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setConfig(prev => ({
                                                        ...prev,
                                                        document_ids: [...prev.document_ids, doc.id]
                                                    }))
                                                } else {
                                                    setConfig(prev => ({
                                                        ...prev,
                                                        document_ids: prev.document_ids.filter(id => id !== doc.id)
                                                    }))
                                                }
                                            }}
                                        />
                                        <div className="flex-1 min-w-0">
                                            <p className="font-medium text-gray-900 truncate">{doc.title}</p>
                                            <p className="text-xs text-gray-500">{doc.num_chunks} chunks</p>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Generation Config */}
                    <div className="card">
                        <div className="flex items-center mb-4">
                            <AdjustmentsHorizontalIcon className="w-5 h-5 text-primary-600 mr-2" />
                            <h2 className="text-lg font-semibold text-gray-900">Cấu hình</h2>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="label">Số lượng câu hỏi</label>
                                <input
                                    type="number"
                                    className="input"
                                    min={1}
                                    max={50}
                                    value={config.num_questions}
                                    onChange={(e) => setConfig(prev => ({
                                        ...prev,
                                        num_questions: parseInt(e.target.value) || 10
                                    }))}
                                />
                            </div>

                            <div>
                                <label className="label">Độ khó</label>
                                <select
                                    className="input"
                                    value={config.difficulty}
                                    onChange={(e) => setConfig(prev => ({
                                        ...prev,
                                        difficulty: e.target.value as any
                                    }))}
                                >
                                    <option value="easy">Dễ</option>
                                    <option value="medium">Trung bình</option>
                                    <option value="hard">Khó</option>
                                </select>
                            </div>

                            <div className="flex items-center">
                                <input
                                    type="checkbox"
                                    id="case_based"
                                    className="rounded border-gray-300 text-primary-600 mr-2"
                                    checked={config.include_case_based}
                                    onChange={(e) => setConfig(prev => ({
                                        ...prev,
                                        include_case_based: e.target.checked
                                    }))}
                                />
                                <label htmlFor="case_based" className="text-sm text-gray-700">
                                    Bao gồm câu hỏi tình huống lâm sàng
                                </label>
                            </div>

                            <div className="flex items-center">
                                <input
                                    type="checkbox"
                                    id="explanations"
                                    className="rounded border-gray-300 text-primary-600 mr-2"
                                    checked={config.include_explanations}
                                    onChange={(e) => setConfig(prev => ({
                                        ...prev,
                                        include_explanations: e.target.checked
                                    }))}
                                />
                                <label htmlFor="explanations" className="text-sm text-gray-700">
                                    Bao gồm giải thích đáp án
                                </label>
                            </div>
                        </div>
                    </div>

                    {/* Generate Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={isGenerating || config.document_ids.length === 0}
                        className="btn-primary w-full flex items-center justify-center"
                    >
                        {isGenerating ? (
                            <span className="flex items-center">
                                <span className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"></span>
                                <span>Đang tạo câu hỏi...</span>
                            </span>
                        ) : (
                            <span className="flex items-center">
                                <SparklesIcon className="w-5 h-5 mr-2" />
                                <span>Tạo câu hỏi</span>
                            </span>
                        )}
                    </button>
                </div>

                {/* Right: Results */}
                <div className="lg:col-span-2">
                    <div className="card">
                        <h2 className="text-lg font-semibold text-gray-900 mb-4">Kết quả</h2>

                        {/* AI Review Stats */}
                        {generationStatus?.status === 'completed' && generationStatus?.review_stats && (
                            <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg border border-blue-200">
                                <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center">
                                    <ShieldCheckIcon className="w-5 h-5 mr-2 text-blue-600" />
                                    Kết quả AI Double-Check
                                </h3>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                    <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                                        <div className="text-xl font-bold text-gray-900">
                                            {generationStatus.review_stats.total_questions}
                                        </div>
                                        <div className="text-xs text-gray-500">Tổng câu hỏi</div>
                                    </div>
                                    <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                                        <div className="text-xl font-bold text-green-600">
                                            {generationStatus.review_stats.high_accuracy}
                                        </div>
                                        <div className="text-xs text-gray-500">Đạt chuẩn</div>
                                    </div>
                                    <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                                        <div className="text-xl font-bold text-yellow-600">
                                            {generationStatus.review_stats.needs_revision}
                                        </div>
                                        <div className="text-xs text-gray-500">Cần sửa</div>
                                    </div>
                                    <div className="text-center p-2 bg-white rounded-lg shadow-sm">
                                        <div className="text-xl font-bold text-blue-600">
                                            {Math.round(generationStatus.review_stats.review_rate * 100)}%
                                        </div>
                                        <div className="text-xs text-gray-500">Đã kiểm tra</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Progress */}
                        {isGenerating && generationStatus && (
                            <div className="mb-6">
                                <div className="flex items-center justify-between text-sm mb-2">
                                    <span className="text-gray-600">Đang tạo câu hỏi...</span>
                                    <span className="text-primary-600 font-medium">
                                        {generationStatus.generated_questions} / {generationStatus.total_questions}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-primary-600 h-2 rounded-full transition-all"
                                        style={{ width: `${generationStatus.progress * 100}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Questions */}
                        {generationStatus?.questions && generationStatus.questions.length > 0 ? (
                            <div className="space-y-4">
                                {generationStatus.questions.map((question, index) => (
                                    <QuestionCard
                                        key={`question-${index}-${question.id || Date.now()}`}
                                        question={question}
                                        questionNumber={index + 1}
                                        mode="preview"
                                        showAnswer={true}
                                        onEdit={handleEditQuestion}
                                    />
                                ))}

                                {/* Actions */}
                                <div className="flex justify-end space-x-4 pt-4 border-t">
                                    <button
                                        onClick={() => navigate('/questions')}
                                        className="btn-secondary"
                                    >
                                        Xem ngân hàng câu hỏi
                                    </button>
                                    <button
                                        onClick={() => {
                                            setGenerationStatus(null)
                                            setConfig(prev => ({ ...prev, document_ids: [] }))
                                        }}
                                        className="btn-primary"
                                    >
                                        Tạo thêm câu hỏi
                                    </button>
                                </div>
                            </div>
                        ) : !isGenerating ? (
                            <div className="text-center py-12">
                                <SparklesIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                                <p className="text-gray-500">
                                    Chọn tài liệu và nhấn "Tạo câu hỏi" để bắt đầu
                                </p>
                            </div>
                        ) : null}
                    </div>
                </div>
            </div>

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
